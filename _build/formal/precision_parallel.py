"""Parallel, resumable precision study for the frozen formal design.

Each worker owns one outer replication and one bootstrap journal, so workers
never write the same file.  The random seeds, estimator and failure handling
are identical to precision.py; this only changes scheduling.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

# Prevent BLAS oversubscription when several Python workers run together.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

import design
import estimation as est
from simulation import synthetic

HERE = Path(__file__).resolve().parent


def read_rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def append(path: Path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_replication(args):
    """Run one outer replication; safe to execute in a spawned process."""
    directory, profile, rho, replication, boots = args
    directory = Path(directory)
    pi = ("weak", "strong").index(profile)
    ri = (0.05, 0.5, 0.95, 1.0).index(float(rho))
    seed = 20261114200 + pi * 1000000 + ri * 100000 + replication
    cell_dir = directory / f"{profile}_rho_{rho:g}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    data = synthetic(seed, rho, profile)
    train = est.subset(data, data["split"] == "train")
    validation = est.subset(data, data["split"] == "validation")
    fitted = est.fit_models(train, validation, models=("RAG_QR",))["RAG_QR"]

    boot_path = cell_dir / f"bootstrap_{replication:04d}.jsonl"
    bootstrap = read_rows(boot_path)
    expected_seeds = [seed * 10000 + i for i in range(boots)]
    if [r["seed"] for r in bootstrap] != expected_seeds[: len(bootstrap)]:
        raise ValueError(f"Bootstrap journal differs from frozen schedule: {boot_path}")
    # Keep the journal resumable while avoiding one open/fsync per bootstrap.
    # A batch is flushed and synced every 25 rows; estimator results and seeds
    # are unchanged, and a killed worker can safely resume from the last line.
    if len(bootstrap) < boots:
        with boot_path.open("a", encoding="utf-8") as stream:
            for b in range(len(bootstrap), boots):
                try:
                    row = est.bootstrap_iteration(train, validation, expected_seeds[b])
                except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                    row = {
                        "seed": expected_seeds[b],
                        "choice_converged": False,
                        "recognition_converged": False,
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                bootstrap.append(row)
                if (b + 1) % 25 == 0 or b == boots - 1:
                    stream.flush()
                    os.fsync(stream.fileno())

    boot_summary = est.summarize_bootstrap(bootstrap, planned=2000)
    interval = boot_summary["rho_percentile_ci_95"]
    success = (
        fitted["choice_fit"]["converged"]
        and fitted["recognition_fit"]["converged"]
        and boot_summary["failure_rate"] <= 0.05
        and interval is not None
    )
    record = {
        "outer": replication,
        "seed": seed,
        "true_rho": rho,
        "rho_hat": fitted["rho"],
        "point_converged": fitted["choice_fit"]["converged"]
        and fitted["recognition_fit"]["converged"],
        "upper_boundary": fitted["beta_at_upper_bound"],
        "lower_boundary": fitted["beta_at_lower_bound"],
        "interval": interval,
        "width": interval[1] - interval[0] if interval else None,
        "covered": bool(interval[0] <= rho <= interval[1]) if interval else False,
        "usable": bool(success),
        "bootstrap_summary": boot_summary,
        "interpretation": "workflow check only"
        if boots < 2000
        else "formal-design precision simulation",
    }
    return str(cell_dir / "outer.jsonl"), record


def _plan(directory: Path, outer: int, boots: int):
    plan = {
        "data_origin": "SYNTHETIC_FORMAL_CANDIDATE_ONLY",
        "real_api_calls": 0,
        "outer_per_cell": outer,
        "scene_bootstraps_per_outer": boots,
        "required_outer_per_cell": 200,
        "required_bootstraps": 2000,
        "profiles": ["weak", "strong"],
        "rho": [0.05, 0.5, 0.95, 1.0],
        "ridges": est.RIDGES,
        "beta_bounds": est.BETA_BOUNDS,
        "estimation": "Fit training only; select ridges on validation marginal NLL; resample each split by scene and reselect both ridges",
        "generation_note": "Independent Bernoulli recognition and choice draws conditional on visible features; no empirical LLM claims",
        "parallel_schedule": "One process per outer replication; deterministic seeds and per-replication journals",
        "hashes": {
            name: design.base.sha((HERE / name).read_bytes())
            for name in ["design.py", "estimation.py", "simulation.py", "precision.py"]
        },
    }
    frozen = directory / "frozen_plan.json"
    if frozen.exists():
        existing = json.loads(frozen.read_text(encoding="utf-8"))
        for key in ("profiles", "rho", "required_outer_per_cell", "required_bootstraps", "ridges", "beta_bounds"):
            # JSON freezes tuples as arrays; compare their values, not Python
            # container types.
            left, right = existing.get(key), plan.get(key)
            if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
                equal = list(left or []) == list(right or [])
            else:
                equal = left == right
            if not equal:
                raise ValueError(f"Existing frozen plan differs at {key}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(design.base.encoded(plan))
    return plan


def _summary(directory: Path, outer: int, boots: int, elapsed: float, plan):
    reports = []
    for profile in plan["profiles"]:
        for rho in plan["rho"]:
            name = f"{profile}_rho_{rho:g}"
            cell_dir = directory / name
            rows = {r["outer"]: r for r in read_rows(cell_dir / "outer.jsonl")}
            if any(i not in rows for i in range(outer)):
                continue
            ordered = [rows[i] for i in range(outer)]
            estimates = np.array([r["rho_hat"] for r in ordered])
            usable = [r for r in ordered if r["usable"]]
            coverage = float(np.mean([r["covered"] for r in usable])) if usable else None
            width = float(np.median([r["width"] for r in usable])) if usable else None
            failure_rate = 1 - len(usable) / outer
            reports.append(
                {
                    "cell": name,
                    "profile": profile,
                    "rho": rho,
                    "outer_completed": outer,
                    "bias": float(estimates.mean() - rho),
                    "rmse": float(np.sqrt(np.mean((estimates - rho) ** 2))),
                    "ci_coverage_successful": coverage,
                    "coverage_all_outer_failures_counted_as_miss": sum(
                        r["covered"] and r["usable"] for r in ordered
                    )
                    / outer,
                    "coverage_mc_se": math.sqrt(coverage * (1 - coverage) / len(usable))
                    if usable
                    else None,
                    "median_ci_width_successful": width,
                    "outer_failure_rate": failure_rate,
                    "upper_boundary_count": sum(r["upper_boundary"] for r in ordered),
                    "lower_boundary_count": sum(r["lower_boundary"] for r in ordered),
                    "numerical_targets_met": coverage is not None
                    and coverage >= 0.90
                    and failure_rate <= 0.05
                    and (rho == 1 or width <= 0.30),
                }
            )
    full = outer >= 200 and boots >= 2000 and len(reports) == 8
    result = {
        "data_origin": plan["data_origin"],
        "real_api_calls": 0,
        "elapsed_seconds": elapsed,
        "full_design_precision_study_complete": full,
        "precision_gate_passed": full and all(r["numerical_targets_met"] for r in reports),
        "formal_ready": False,
        "cells": reports,
        "scope": "Precision gate only; technical pilot and human measurement checks remain separate",
    }
    (directory / "summary.json").write_bytes(design.base.encoded(result))
    return result


def run(directory: Path, outer: int, boots: int, workers: int):
    if outer < 1 or boots < 2:
        raise ValueError("outer>=1, boots>=2 required")
    directory.mkdir(parents=True, exist_ok=True)
    plan = _plan(directory, outer, boots)
    start = time.monotonic()
    tasks = []
    for profile in plan["profiles"]:
        for rho in plan["rho"]:
            cell_dir = directory / f"{profile}_rho_{rho:g}"
            completed = {r["outer"] for r in read_rows(cell_dir / "outer.jsonl")}
            for replication in range(outer):
                if replication not in completed:
                    tasks.append((str(directory), profile, rho, replication, boots))
    print(json.dumps({"pending_replications": len(tasks), "workers": workers}), flush=True)
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_replication, task) for task in tasks]
            for future in as_completed(futures):
                path, record = future.result()
                append(Path(path), record)
                print(
                    json.dumps(
                        {
                            "cell": Path(path).parent.name,
                            "outer": record["outer"],
                            "rho_hat": record["rho_hat"],
                            "ci": record["interval"],
                            "covered": record["covered"],
                            "boots": boots,
                        }
                    ),
                    flush=True,
                )
    return _summary(directory, outer, boots, time.monotonic() - start, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "precision_parallel_check")
    parser.add_argument("--outer", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--run-full", action="store_true")
    args = parser.parse_args()
    if args.run_full:
        args.outer, args.bootstrap = 200, 2000
        if args.out == HERE / "precision_parallel_check":
            args.out = HERE / "precision_full"
    result = run(args.out, args.outer, args.bootstrap, args.workers)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
