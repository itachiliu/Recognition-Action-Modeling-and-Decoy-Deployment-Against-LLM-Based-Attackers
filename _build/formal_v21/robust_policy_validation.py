"""Offline validation of model-robust deployment policies on v1.0 data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_build" / "formal"))
sys.path.insert(0, str(HERE := ROOT / "_build" / "formal_v21"))

import deployment as dep

CLOSED = HERE / "closed_loop_v2"
MAIN = HERE / "main_v21"
OUTPUT = HERE / "robust_policy_validation.json"
MODELS = ("Deepseek-Flash", "Deepseek-R1-32b", "GLM-5")
RESPONSE_MODELS = ("rational", "QR", "SUQR", "fixed_rho", "RAG_QR")


def load(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def plan_dict(plan):
    return {int(key): tuple(value) for key, value in plan.items()}


def actual_metrics(rows):
    values = []
    for row in rows:
        values.append(3.0 if row["canonical_choice"].startswith("D") else -row["canonical_truth"][row["canonical_choice"]]["loss"])
    return float(np.mean(values)) if values else None


def predicted_utilities(scene, plan, fitted):
    utilities = {}
    for name in RESPONSE_MODELS:
        if name == "rational":
            probabilities = dep.predicted_probabilities(scene, plan, name, fitted[name])
        elif name == "QR":
            probabilities = dep.predicted_probabilities(scene, plan, name, fitted[name])
        else:
            probabilities = dep.predicted_probabilities(scene, plan, name, fitted[name])
        targets = dep.materialize(scene, plan)
        value = sum(
            3.0 * probability if target["truth"]["is_decoy"] else -target["truth"]["business_loss_value"] * probability
            for probability, target in zip(probabilities, targets)
        )
        utilities[name] = value
    return utilities


def choose_robust(scene_plans, scene, fitted, rule):
    scores = []
    for row in scene_plans:
        utilities = predicted_utilities(scene, plan_dict(row["plan"]), fitted)
        score = min(utilities.values()) if rule == "min" else float(np.mean(list(utilities.values())))
        scores.append((score, row["strategy"]))
    return max(scores, key=lambda item: (item[0], -dep.STRATEGIES.index(item[1])))[1]


def scene_bootstrap_difference(reference, candidate, seed=20261118201, repetitions=5000):
    common = [scene for scene in reference if scene in candidate and reference[scene] is not None and candidate[scene] is not None]
    if not common:
        return None
    differences = np.array([candidate[scene] - reference[scene] for scene in common])
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(0, len(differences), (repetitions, len(differences)))].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci_95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "scenes": len(common),
        "p_two_sided_centered": float((1 + np.count_nonzero(np.abs(sampled - differences.mean()) >= abs(differences.mean()))) / (repetitions + 1)),
    }


def run():
    plans = load(CLOSED / "plans.jsonl")
    tasks = load(CLOSED / "tasks.jsonl")
    raw = load(CLOSED / "raw.jsonl")
    lookup = {task["task_id"]: task for task in tasks}
    result = {"models": {}, "adoption": {}}

    for model in MODELS:
        fitted = json.loads((MAIN / "analysis" / model / "fitted_models.json").read_text(encoding="utf-8"))
        model_plans = [row for row in plans if row["model"] == model]
        scenes = sorted({row["scene_id"] for row in model_plans})
        per_scene = {}
        for scene_id in scenes:
            scene = dep.make_scene(int(scene_id[3:]) - 1)
            rows = [row for row in model_plans if row["scene_id"] == scene_id]
            per_scene[scene_id] = {row["strategy"]: row for row in rows}

        actual_by_strategy = {}
        for strategy in dep.STRATEGIES:
            values = {}
            for scene_id in scenes:
                scene_rows = [
                    row for row in raw
                    if row["model"] == model and row["scene_id"] == scene_id and row["strategy"] == strategy and row["ok"]
                ]
                if scene_rows:
                    scene_values = []
                    for row in scene_rows:
                        task = lookup[row["task_id"]]
                        truth = task["canonical_truth"][row["canonical_choice"]]
                        scene_values.append(3.0 if truth["is_decoy"] else -truth["loss"])
                    values[scene_id] = float(np.mean(scene_values))
            actual_by_strategy[strategy] = values

        model_result = {"rules": {}}
        for rule in ("min", "mean"):
            selected = {}
            for scene_id in scenes:
                scene = dep.make_scene(int(scene_id[3:]) - 1)
                scene_plans = per_scene[scene_id]
                selected[scene_id] = choose_robust(
                    list(scene_plans.values()), scene, fitted, rule
                )
            robust_actual = {
                scene_id: actual_by_strategy[selected[scene_id]].get(scene_id)
                for scene_id in scenes
            }
            model_result["rules"][rule] = {
                "selected_strategies": selected,
                "actual_mean_utility": float(np.mean([value for value in robust_actual.values() if value is not None])),
                "vs_suqr": scene_bootstrap_difference(actual_by_strategy["SUQR"], robust_actual),
                "vs_rag": scene_bootstrap_difference(actual_by_strategy["RAG_QR"], robust_actual),
                "vs_best_single": scene_bootstrap_difference(
                    {scene: max((actual_by_strategy[s].get(scene) if actual_by_strategy[s].get(scene) is not None else -np.inf) for s in dep.STRATEGIES) for scene in scenes},
                    robust_actual,
                ),
            }
        result["models"][model] = model_result

    adoption = {}
    for rule in ("min", "mean"):
        positive_suqr = sum(
            result["models"][model]["rules"][rule]["vs_suqr"]["ci_95"][0] > 0
            for model in MODELS
        )
        positive_rag = sum(
            result["models"][model]["rules"][rule]["vs_rag"]["ci_95"][0] > 0
            for model in MODELS
        )
        adoption[rule] = {
            "adopt": positive_suqr >= 2 and positive_rag >= 2,
            "positive_vs_suqr_models": positive_suqr,
            "positive_vs_rag_models": positive_rag,
            "criterion": "At least two models must show a 95% bootstrap interval above zero versus both SUQR and RAG-QR.",
        }
    result["adoption"] = adoption
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
