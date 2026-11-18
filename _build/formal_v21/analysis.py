"""Formal-v2.1 analysis of the completed real-LLM collection."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_build" / "formal"))

import estimation as est
from interval_diagnostic import cluster_robust_choice_covariance
from profile_interval_diagnostic import optimize_gamma, profile_end

HERE = Path(__file__).resolve().parent
RUN = HERE / "main_v21"


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def auc_binary(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return None
    comparisons = scores[positives, None] - scores[None, negatives]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def analyze_model(model, presentations, scenes, records):
    split_by_scene = {scene["scene_id"]: scene["split"] for scene in scenes}
    valid_records = [record for record in records if record["model"] == model and record["ok"]]
    data = est.dataset_from_records(presentations, valid_records, split_by_scene)
    train = est.subset(data, data["split"] == "train")
    validation = est.subset(data, data["split"] == "validation")
    test = est.subset(data, data["split"] == "test")
    fitted = est.fit_models(train, validation)
    metrics = {name: est.prediction_metrics(test, value) for name, value in fitted.items()}

    rag = fitted["RAG_QR"]
    gamma_hat = np.asarray(rag["gamma"])
    beta_hat = float(rag["beta"])
    _, center_loss = optimize_gamma(train, beta_hat, rag["choice_ridge"], gamma_hat)
    lower_beta, _, _ = profile_end(
        train, rag["choice_ridge"], beta_hat, gamma_hat, center_loss, -1, est.BETA_BOUNDS[0]
    )
    upper_beta, _, _ = profile_end(
        train, rag["choice_ridge"], beta_hat, gamma_hat, center_loss, 1, est.BETA_BOUNDS[1]
    )
    lower_rho = 0.0 if lower_beta <= est.BETA_BOUNDS[0] + 1e-7 else math.exp(lower_beta)
    upper_rho = 1.0 if upper_beta >= est.BETA_BOUNDS[1] - 1e-7 else math.exp(upper_beta)
    covariance = cluster_robust_choice_covariance(
        train, gamma_hat, beta_hat, rag["choice_ridge"]
    )
    beta_se = float(math.sqrt(max(covariance[-1, -1], 0.0)))

    z = est.recognition_design(test)[test["mask"]]
    labels = test["r"][test["mask"]]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(z @ np.asarray(rag["alpha"]), -700, 700)))
    comparisons = {}
    for baseline in ("rational", "QR", "SUQR", "fixed_rho"):
        comparisons[f"RAG_QR_minus_{baseline}"] = est.paired_scene_comparison(
            metrics["RAG_QR"], metrics[baseline], repetitions=2000
        )
    p_values = [value["bootstrap_p_two_sided_centered"] for value in comparisons.values()]
    adjusted = est.holm_adjust(p_values)
    for value, adjusted_p in zip(comparisons.values(), adjusted):
        value["holm_adjusted_p"] = adjusted_p

    chosen_r = test["r"][np.arange(len(test["choice"])), test["choice"]]
    return {
        "model": model,
        "records": {
            "planned": 3072,
            "completed": sum(record["model"] == model for record in records),
            "valid": len(valid_records),
        },
        "data": {
            "train_choices": len(train["choice"]),
            "validation_choices": len(validation["choice"]),
            "test_choices": len(test["choice"]),
            "recognition_positive_rate": float(data["r"][data["mask"]].mean()),
            "test_chosen_marked_rate": float(chosen_r.mean()),
            "design_diagnostics": {
                split_name: est.design_diagnostics(part)
                for split_name, part in [
                    ("train", train),
                    ("validation", validation),
                    ("test", test),
                ]
            },
        },
        "rho": {
            "estimate": float(rag["rho"]),
            "profile_95": [lower_rho, upper_rho],
            "beta": beta_hat,
            "profile_beta_95": [lower_beta, upper_beta],
            "cluster_robust_beta_se": beta_se,
            "choice_ridge": rag["choice_ridge"],
            "recognition_ridge": rag["recognition_ridge"],
            "upper_boundary": rag["beta_at_upper_bound"],
            "lower_boundary": rag["beta_at_lower_bound"],
        },
        "recognition": {
            "coefficients": dict(zip(["intercept", "z1", "z2", "z3", "z4"], rag["alpha"])),
            "test_auc": auc_binary(labels, probabilities),
            "test_brier": float(np.mean((probabilities - labels) ** 2)),
        },
        "prediction": {
            name: {
                key: value[key]
                for key in ("nll", "top1", "brier", "top_label_ece_10_bins")
            }
            for name, value in metrics.items()
        },
        "comparisons": comparisons,
        "fits": fitted,
    }


def main():
    presentations = load_jsonl(RUN / "presentations.jsonl")
    scenes = load_jsonl(RUN / "dataset_v2.jsonl")
    records = load_jsonl(RUN / "raw.jsonl")
    config = json.loads((RUN / "frozen_config.json").read_text(encoding="utf-8"))
    results = {
        "protocol_version": config["protocol_version"],
        "planned_tasks": 9216,
        "completed_tasks": len(records),
        "duplicate_task_ids": len(records) - len({record["task_id"] for record in records}),
        "models": {},
    }
    for model in config["models"]:
        result = analyze_model(model, presentations, scenes, records)
        fitted = result.pop("fits")
        output_dir = RUN / "analysis" / model
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fitted_models.json").write_text(
            json.dumps(fitted, ensure_ascii=False, indent=2, default=list) + "\n",
            encoding="utf-8",
        )
        results["models"][model] = result
        print(
            json.dumps(
                {
                    "model": model,
                    "rho": result["rho"]["estimate"],
                    "rho_ci": result["rho"]["profile_95"],
                    "rag_nll": result["prediction"]["RAG_QR"]["nll"],
                    "suqr_nll": result["prediction"]["SUQR"]["nll"],
                }
            ),
            flush=True,
        )
    (RUN / "formal_analysis.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    main()
