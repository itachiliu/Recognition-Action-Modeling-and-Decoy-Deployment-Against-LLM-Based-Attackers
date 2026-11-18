"""Resumable formal-design coverage study; default is a small workflow check.

The full gate needs 8 cells x 200 outer datasets x 2000 scene bootstraps.
Small execution is explicitly incapable of declaring the design ready.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import design
import estimation as est
from simulation import synthetic

HERE = Path(__file__).resolve().parent


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(directory, outer=2, boots=20):
    if outer < 1 or boots < 2:
        raise ValueError("outer>=1, boots>=2 required")
    directory.mkdir(parents=True, exist_ok=True)
    plan = {
        "data_origin": "SYNTHETIC_FORMAL_CANDIDATE_ONLY", "real_api_calls": 0,
        "outer_per_cell": outer, "scene_bootstraps_per_outer": boots,
        "required_outer_per_cell": 200, "required_bootstraps": 2000,
        "profiles": ["weak", "strong"], "rho": [.05, .5, .95, 1.],
        "ridges": est.RIDGES, "beta_bounds": est.BETA_BOUNDS,
        "estimation": "Fit training only; select ridges on validation marginal NLL; resample each split by scene and reselect both ridges",
        "generation_note": "Independent Bernoulli recognition and choice draws conditional on visible features; no empirical LLM claims",
        "hashes": {name: design.base.sha((HERE / name).read_bytes())
                   for name in ["design.py", "estimation.py", "simulation.py", "precision.py"]},
    }
    design.base.immutable(directory / "frozen_plan.json", design.base.encoded(plan))
    for name in plan["hashes"]:
        design.base.immutable(directory / ("snapshot_" + name), (HERE / name).read_bytes())
    total_start = time.monotonic()
    reports = []
    for pi, profile in enumerate(plan["profiles"]):
        for ri, rho in enumerate(plan["rho"]):
            name = f"{profile}_rho_{rho:g}"
            cell_dir = directory / name
            cell_dir.mkdir(exist_ok=True)
            completed = {r["outer"]: r for r in read_rows(cell_dir / "outer.jsonl")}
            for replication in range(outer):
                if replication in completed:
                    continue
                seed = 20261114200 + pi * 1000000 + ri * 100000 + replication
                data = synthetic(seed, rho, profile)
                train = est.subset(data, data["split"] == "train")
                validation = est.subset(data, data["split"] == "validation")
                fitted = est.fit_models(train, validation, models=("RAG_QR",))["RAG_QR"]
                boot_path = cell_dir / f"bootstrap_{replication:04d}.jsonl"
                bootstrap = read_rows(boot_path)
                expected_seeds = [seed * 10000 + i for i in range(boots)]
                if [r["seed"] for r in bootstrap] != expected_seeds[:len(bootstrap)]:
                    raise ValueError("Bootstrap journal differs from frozen schedule")
                for b in range(len(bootstrap), boots):
                    try:
                        row = est.bootstrap_iteration(train, validation, expected_seeds[b])
                    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                        row = {"seed": expected_seeds[b], "choice_converged": False,
                               "recognition_converged": False, "error": type(exc).__name__ + ": " + str(exc)}
                    append(boot_path, row)
                    bootstrap.append(row)
                boot_summary = est.summarize_bootstrap(bootstrap, planned=2000)
                interval = boot_summary["rho_percentile_ci_95"]
                success = (fitted["choice_fit"]["converged"] and fitted["recognition_fit"]["converged"]
                           and boot_summary["failure_rate"] <= .05 and interval is not None)
                record = {
                    "outer": replication, "seed": seed, "true_rho": rho, "rho_hat": fitted["rho"],
                    "point_converged": fitted["choice_fit"]["converged"] and fitted["recognition_fit"]["converged"],
                    "upper_boundary": fitted["beta_at_upper_bound"], "lower_boundary": fitted["beta_at_lower_bound"],
                    "interval": interval, "width": interval[1] - interval[0] if interval else None,
                    "covered": bool(interval[0] <= rho <= interval[1]) if interval else False,
                    "usable": bool(success), "bootstrap_summary": boot_summary,
                    "interpretation": "workflow check only" if outer < 200 or boots < 2000 else "formal-design precision simulation",
                }
                append(cell_dir / "outer.jsonl", record)
                completed[replication] = record
                print(json.dumps({"cell": name, "outer": replication, "rho_hat": record["rho_hat"],
                                  "ci": interval, "covered": record["covered"], "boots": boots}), flush=True)
            rows = [completed[i] for i in range(outer)]
            estimates = np.array([r["rho_hat"] for r in rows])
            usable = [r for r in rows if r["usable"]]
            coverage = float(np.mean([r["covered"] for r in usable])) if usable else None
            width = float(np.median([r["width"] for r in usable])) if usable else None
            failure_rate = 1 - len(usable) / outer
            reports.append({
                "cell": name, "profile": profile, "rho": rho, "outer_completed": outer,
                "bias": float(estimates.mean() - rho), "rmse": float(np.sqrt(np.mean((estimates - rho) ** 2))),
                "ci_coverage_successful": coverage,
                "coverage_all_outer_failures_counted_as_miss": sum(r["covered"] and r["usable"] for r in rows) / outer,
                "coverage_mc_se": math.sqrt(coverage * (1 - coverage) / len(usable)) if usable else None,
                "median_ci_width_successful": width, "outer_failure_rate": failure_rate,
                "upper_boundary_count": sum(r["upper_boundary"] for r in rows),
                "lower_boundary_count": sum(r["lower_boundary"] for r in rows),
                "numerical_targets_met": coverage is not None and coverage >= .90 and failure_rate <= .05
                                         and (rho == 1 or width <= .30),
            })
    full = outer >= 200 and boots >= 2000
    result = {
        "data_origin": plan["data_origin"], "real_api_calls": 0,
        "elapsed_seconds": time.monotonic() - total_start,
        "full_design_precision_study_complete": full,
        "precision_gate_passed": full and all(r["numerical_targets_met"] for r in reports),
        "formal_ready": False, "cells": reports,
        "scope": "Precision gate only; technical pilot and human measurement checks remain separate",
    }
    (directory / "summary.json").write_bytes(design.base.encoded(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "precision_workflow_check_v2")
    parser.add_argument("--outer", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--run-full", action="store_true", help="Long offline run: exactly 200 outer and 2000 bootstrap replicates.")
    args = parser.parse_args()
    if args.run_full:
        args.outer, args.bootstrap = 200, 2000
        if args.out == HERE / "precision_workflow_check_v2":
            args.out = HERE / "precision_full"
    result = run(args.out, args.outer, args.bootstrap)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
