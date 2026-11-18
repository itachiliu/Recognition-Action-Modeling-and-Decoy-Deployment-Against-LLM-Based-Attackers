"""Validate the soft-recognition variant of RAG-QR on the frozen test set."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_build" / "formal"))

import estimation as est

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main_v21"
OUTPUT = HERE / "variant_validation_soft.json"
MODELS = ("Deepseek-Flash", "Deepseek-R1-32b", "GLM-5")


def load(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def soft_features(data, alpha):
    z = est.recognition_design(data)
    probability = 1 / (1 + np.exp(-np.clip(z @ np.asarray(alpha), -700, 700)))
    return np.concatenate([data["F"], probability[..., None]], axis=-1), probability


def soft_probabilities(data, fitted):
    features, probability = soft_features(data, fitted["alpha"])
    theta = np.r_[np.asarray(fitted["gamma"]), float(fitted["beta"])]
    return est.probabilities(features @ theta, data["mask"])


def soft_nll(data, fitted):
    probabilities = soft_probabilities(data, fitted)
    selected = probabilities[np.arange(len(probabilities)), data["choice"]]
    return float(-np.log(np.maximum(selected, 1e-300)).mean())


def soft_metrics(data, fitted):
    probabilities = soft_probabilities(data, fitted)
    selected = probabilities[np.arange(len(probabilities)), data["choice"]]
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == data["choice"]).astype(float)
    brier = (probabilities * probabilities).sum(axis=1) - 2 * selected + 1
    ece = 0.0
    for bucket in range(10):
        lower = bucket / 10
        upper = (bucket + 1) / 10
        members = (confidence >= lower) & (confidence < upper)
        if members.any():
            ece += members.mean() * abs(correct[members].mean() - confidence[members].mean())
    per_scene = {}
    for scene in np.unique(data["scene"]):
        rows = data["scene"] == scene
        per_scene[str(scene)] = {
            "nll": float(-np.log(np.maximum(selected[rows], 1e-300)).mean()),
            "top1": float(correct[rows].mean()),
            "brier": float(brier[rows].mean()),
            "valid_tasks": int(rows.sum()),
        }
    return {
        "nll": float(-np.log(np.maximum(selected, 1e-300)).mean()),
        "top1": float(correct.mean()),
        "brier": float(brier.mean()),
        "top_label_ece_10_bins": float(ece),
        "per_scene": per_scene,
    }


def fit_soft(train, validation, alpha):
    base = {
        "model": "RAG_QR_soft",
        "alpha": list(alpha),
        "recognition_ridge": None,
    }
    candidates = []
    for ridge in est.RIDGES:
        data = {
            key: value
            for key, value in train.items()
        }
        features, probability = soft_features(train, alpha)
        data["r"] = probability
        gamma, beta, status = est.fit_choice(data, ridge, np.ones(len(data["choice"])), "RAG_QR")
        fitted = {
            **base,
            "gamma": list(gamma),
            "beta": float(beta),
            "choice_ridge": ridge,
            "choice_fit": status,
        }
        candidates.append((soft_nll(validation, fitted), ridge, fitted))
    _, _, selected = min(candidates, key=lambda value: value[0])
    return selected


def run():
    presentations = load(MAIN / "presentations.jsonl")
    scenes = load(MAIN / "dataset_v2.jsonl")
    records = load(MAIN / "raw.jsonl")
    split_by_scene = {scene["scene_id"]: scene["split"] for scene in scenes}
    reports = []
    adoptions = []
    for model in MODELS:
        valid = [record for record in records if record["model"] == model and record["ok"]]
        data = est.dataset_from_records(presentations, valid, split_by_scene)
        train = est.subset(data, data["split"] == "train")
        validation = est.subset(data, data["split"] == "validation")
        test = est.subset(data, data["split"] == "test")
        fitted_all = est.fit_models(train, validation)
        original_rag = fitted_all["RAG_QR"]
        original_metrics = est.prediction_metrics(test, original_rag)
        suqr_metrics = est.prediction_metrics(test, fitted_all["SUQR"])
        soft = fit_soft(train, validation, original_rag["alpha"])
        soft_result = soft_metrics(test, soft)

        vs_original = est.paired_scene_comparison(
            soft_result, original_metrics, repetitions=2000
        )
        vs_suqr = est.paired_scene_comparison(
            soft_result, suqr_metrics, repetitions=2000
        )
        relative_change = (
            (soft_result["nll"] - original_metrics["nll"]) / original_metrics["nll"]
        )
        report = {
            "model": model,
            "original_nll": original_metrics["nll"],
            "suqr_nll": suqr_metrics["nll"],
            "soft_nll": soft_result["nll"],
            "soft_top1": soft_result["top1"],
            "soft_ece": soft_result["top_label_ece_10_bins"],
            "relative_change_vs_original": relative_change,
            "soft_minus_original": vs_original,
            "soft_minus_suqr": vs_suqr,
            "choice_ridge": soft["choice_ridge"],
        }
        reports.append(report)
        adoptions.append(
            {
                "model": model,
                "meaningful_improvement": relative_change <= -0.05,
                "worse": relative_change >= 0.01,
            }
        )
    meaningful_count = sum(item["meaningful_improvement"] for item in adoptions)
    any_worse = any(item["worse"] for item in adoptions)
    adopt = meaningful_count >= 2 and not any_worse
    result = {
        "data_origin": "FROZEN_MAIN_TEST_SPLIT",
        "pre_specified_gate": (
            "Adopt soft-recognition variant only if at least two models improve test "
            "NLL by at least 5% relative to the original RAG-QR, and no model worsens "
            "by more than 1%."
        ),
        "models": reports,
        "meaningful_improvement_count": meaningful_count,
        "any_model_worse": any_worse,
        "adopt_variant": adopt,
        "decision": "ADOPT" if adopt else "DO_NOT_ADOPT",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
