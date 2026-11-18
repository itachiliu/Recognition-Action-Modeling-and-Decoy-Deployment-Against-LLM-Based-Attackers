"""Offline model-based budget sensitivity for formal-v2.1."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import deployment as dep

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main_v21"
OUTPUT = HERE / "deployment_sensitivity.json"
BUDGETS = (4, 6, 8, 10, 12, 16)


def metrics(scene, plan, strategy, fitted):
    probabilities = dep.predicted_probabilities(scene, plan, strategy, fitted)
    targets = dep.materialize(scene, plan)
    hit = sum(
        probability
        for probability, target in zip(probabilities, targets)
        if target["truth"]["is_decoy"]
    )
    loss = sum(
        probability * target["truth"]["business_loss_value"]
        for probability, target in zip(probabilities, targets)
        if not target["truth"]["is_decoy"]
    )
    cost = sum(dep.config_cost(config) for config in plan.values())
    high_cost = sum(
        dep.LEVEL_COST[config[0]]
        for config in plan.values()
        if config[0] == "high"
    )
    return {
        "hit_rate": hit,
        "expected_loss": loss,
        "utility": 3 * hit - loss,
        "cost": cost,
        "decoys": len(plan),
        "high_realism_cost": high_cost,
        "high_realism_share": high_cost / cost if cost else 0.0,
    }


def main():
    analysis = json.loads((MAIN / "formal_analysis.json").read_text(encoding="utf-8"))
    rows = []
    for model in analysis["models"]:
        fitted = json.loads((MAIN / "analysis" / model / "fitted_models.json").read_text(encoding="utf-8"))
        for scene_index in range(20):
            scene = dep.make_scene(scene_index)
            for budget in BUDGETS:
                for strategy in ("SUQR", "RAG_QR"):
                    plan, _ = dep.optimize_plan(
                        scene, strategy, fitted[strategy], budget=budget
                    )
                    value = metrics(scene, plan, strategy, fitted[strategy])
                    rows.append(
                        {
                            "model": model,
                            "scene_id": scene["scene_id"],
                            "budget": budget,
                            "strategy": strategy,
                            **value,
                        }
                    )
    summary = {}
    for model in analysis["models"]:
        summary[model] = {}
        for budget in BUDGETS:
            summary[model][str(budget)] = {}
            for strategy in ("SUQR", "RAG_QR"):
                subset = [
                    row for row in rows
                    if row["model"] == model
                    and row["budget"] == budget
                    and row["strategy"] == strategy
                ]
                summary[model][str(budget)][strategy] = {
                    key: statistics.mean(row[key] for row in subset)
                    for key in (
                        "hit_rate",
                        "expected_loss",
                        "utility",
                        "cost",
                        "decoys",
                        "high_realism_share",
                    )
                }
    result = {
        "data_origin": "MODEL_PREDICTION_NOT_CLOSED_LOOP",
        "budgets": list(BUDGETS),
        "rows": rows,
        "summary": summary,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
