"""Small end-to-end offline demonstration, explicitly not paper results."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import design
import estimation as est
from simulation import synthetic

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=40)
    parser.add_argument("--out", type=Path, default=HERE / "synthetic_analysis_check")
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be >=2")
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        "data_origin": "SYNTHETIC_ONLY_NOT_LLM_RESULTS", "seed": 20261114122,
        "rho_true": .5, "profile": "strong", "scene_bootstrap_repetitions": args.bootstrap,
    }
    design.base.immutable(args.out / "frozen_config.json", design.base.encoded(config))
    for name in ["analysis_smoke.py", "design.py", "simulation.py", "estimation.py"]:
        design.base.immutable(args.out / ("snapshot_" + name), (HERE / name).read_bytes())
    data = synthetic(config["seed"], config["rho_true"], config["profile"])
    partitions = {key: est.subset(data, data["split"] == key) for key in ("train", "validation", "test")}
    fitted = est.fit_models(partitions["train"], partitions["validation"])
    metrics = {name: est.prediction_metrics(partitions["test"], fit) for name, fit in fitted.items()}
    comparisons = {
        name: est.paired_scene_comparison(metrics["RAG_QR"], metrics[name])
        for name in est.MODELS if name != "RAG_QR"
    }
    p_adjusted = est.holm_adjust([r["bootstrap_p_two_sided_centered"] for r in comparisons.values()])
    for value, p in zip(comparisons.values(), p_adjusted):
        value["holm_adjusted_p"] = p
    path = args.out / "bootstrap.jsonl"
    bootstrap = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    seeds = [20261114600 + i for i in range(args.bootstrap)]
    if [r["seed"] for r in bootstrap] != seeds[:len(bootstrap)]:
        raise ValueError("Unexpected bootstrap seed/order")
    with path.open("a", encoding="utf-8") as stream:
        for i in range(len(bootstrap), args.bootstrap):
            try:
                row = est.bootstrap_iteration(partitions["train"], partitions["validation"], seeds[i])
            except (ValueError, np.linalg.LinAlgError) as exc:
                row = {"seed": seeds[i], "choice_converged": False, "recognition_converged": False,
                       "error": type(exc).__name__ + ": " + str(exc)}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            bootstrap.append(row)
    report = {
        **config, "real_api_calls": 0, "formal_ready": False,
        "fits": fitted, "held_out_marginal_metrics": metrics, "comparisons": comparisons,
        "rho_bootstrap": est.summarize_bootstrap(bootstrap),
        "training_diagnostics": est.design_diagnostics(partitions["train"]),
        "not_completed": [
            "Human measurement review", "Full formal-design precision study", "Actual LLM model estimation",
            "Profile likelihood / separation and numerical-lower-bound sensitivity", "Recognition AUC / clustered standard errors",
            "Null parametric bootstrap for optional boundary p-values",
            "Deployment optimizer and independent closed-loop evaluation",
        ],
        "scope": "Synthetic workflow demonstration, not comparison results suitable for the article",
    }
    (args.out / "summary.json").write_bytes(design.base.encoded(report))
    print(json.dumps({
        "origin": config["data_origin"], "bootstrap": report["rho_bootstrap"],
        "held_out_nll": {name: m["nll"] for name, m in metrics.items()},
        "output": str(args.out),
    }, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
