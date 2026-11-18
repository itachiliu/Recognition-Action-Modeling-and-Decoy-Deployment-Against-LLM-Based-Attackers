"""Large-scale independent validation of the model-robust deployment policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_build" / "formal"))
sys.path.insert(0, str(ROOT / "_build" / "formal_v21"))
sys.path.insert(0, str(ROOT / "_build" / "v25"))

import deployment as dep
import runner as transport

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main_v21"
OUTPUT = HERE / "robust_large_scale"
MODELS = ("Deepseek-Flash", "Deepseek-R1-32b", "GLM-5")
RESPONSE_MODELS = ("rational", "QR", "SUQR", "fixed_rho", "RAG_QR")
SCENE_SEED = 20261119001


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jsonlines(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen file differs: {path}")
    else:
        path.write_bytes(data)


def load(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def plan_dict(plan):
    return {int(key): tuple(value) for key, value in plan.items()}


def predicted_utility(scene, plan, fitted):
    probabilities = dep.predicted_probabilities(scene, plan, "RAG_QR", fitted)
    targets = dep.materialize(scene, plan)
    return sum(
        3.0 * probability if target["truth"]["is_decoy"] else -target["truth"]["business_loss_value"] * probability
        for probability, target in zip(probabilities, targets)
    )


def robust_score(scene, plan, fitted, rule="min"):
    values = []
    for name in RESPONSE_MODELS:
        probabilities = dep.predicted_probabilities(scene, plan, name, fitted[name])
        targets = dep.materialize(scene, plan)
        utility = sum(
            3.0 * probability if target["truth"]["is_decoy"] else -target["truth"]["business_loss_value"] * probability
            for probability, target in zip(probabilities, targets)
        )
        values.append(utility)
    return min(values) if rule == "min" else float(np.mean(values))


def choose_strategy(scene, strategy_plans, fitted, rule="min"):
    best = None
    for row in strategy_plans:
        score = robust_score(scene, plan_dict(row["plan"]), fitted, rule)
        key = (score, -dep.STRATEGIES.index(row["strategy"]))
        if best is None or key > best[0]:
            best = (key, row)
    return best[1]


def prepare(scenes=60):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((MAIN / "formal_analysis.json").read_text(encoding="utf-8"))
    tasks = []
    records = []
    for model in MODELS:
        fitted = json.loads((MAIN / "analysis" / model / "fitted_models.json").read_text(encoding="utf-8"))
        for scene_index in range(scenes):
            scene = dep.make_scene(scene_index, SCENE_SEED)
            strategy_plans = []
            for strategy in dep.STRATEGIES:
                if strategy in ("random", "all_high", "max_cue"):
                    plan = dep.heuristic_plan(scene, strategy, SCENE_SEED + scene_index)
                else:
                    plan, _ = dep.optimize_plan(scene, strategy, fitted[strategy])
                strategy_plans.append({"strategy": strategy, "plan": {str(k): list(v) for k, v in plan.items()}})
            robust = choose_strategy(scene, strategy_plans, fitted, "min")
            plans = {
                "robust": robust,
                "SUQR": next(row for row in strategy_plans if row["strategy"] == "SUQR"),
                "RAG_QR": next(row for row in strategy_plans if row["strategy"] == "RAG_QR"),
            }
            for arm, row in plans.items():
                targets = dep.materialize(scene, plan_dict(row["plan"]))
                record = {
                    "model": model,
                    "scene_id": scene["scene_id"],
                    "arm": arm,
                    "selected_strategy": row["strategy"] if arm == "robust" else arm,
                    "plan": row["plan"],
                    "cost": sum(dep.config_cost(config) for config in plan_dict(row["plan"]).values()),
                }
                records.append(record)
                for item in dep.build_presentations(scene, plan_dict(row["plan"])):
                    for repeat in range(4):
                        tasks.append({
                            "task_id": f"{model}|{scene['scene_id']}|{arm}|{item['layout']}|{repeat}",
                            "model": model,
                            "scene_id": scene["scene_id"],
                            "arm": arm,
                            "selected_strategy": record["selected_strategy"],
                            "layout": item["layout"],
                            "repeat": repeat,
                            "layout_seed": item["layout_seed"],
                            "prompt": item["prompt"],
                            "prompt_sha256": item["prompt_sha256"],
                            "display_to_canonical": item["display_to_canonical"],
                            "canonical_truth": {
                                target["canonical_id"]: {
                                    "is_decoy": target["truth"]["is_decoy"],
                                    "loss": target["truth"]["business_loss_value"],
                                }
                                for target in targets
                            },
                        })
    random.Random(20261119099).shuffle(tasks)
    immutable(OUTPUT / "plans.jsonl", jsonlines(records).encode("utf-8"))
    immutable(OUTPUT / "tasks.jsonl", jsonlines(tasks).encode("utf-8"))
    audit = {
        "models": list(MODELS),
        "scenes": scenes,
        "arms": ["robust", "SUQR", "RAG_QR"],
        "planned_tasks": len(tasks),
        "duplicate_task_ids": len(tasks) - len({task["task_id"] for task in tasks}),
        "formal_ready": len(tasks) == scenes * len(MODELS) * 3 * 8,
    }
    (OUTPUT / "design_audit.json").write_bytes(encoded(audit))
    return records, tasks, audit


def run_api(tasks, limit=None):
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    credential = json.loads((ROOT / "_build" / "api_config.local.json").read_text(encoding="utf-8"))
    key = os.environ.get("LLM_API_KEY", "").strip() or str(credential.get("api_key", "")).strip()
    if not key:
        raise SystemExit("No API key")
    raw_path = OUTPUT / "raw.jsonl"
    existing = load(raw_path) if raw_path.exists() else []
    done = {row["task_id"] for row in existing}
    pending = [task for task in tasks if task["task_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    config_hash = sha(encoded(config))

    def one(task):
        payload = {
            "model": task["model"],
            "messages": [{"role": "user", "content": task["prompt"]}],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "max_tokens": config["max_tokens_by_model"][task["model"]],
        }
        if task["model"] in config["disable_thinking_request_models"]:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.update(config.get("request_overrides", {}).get(task["model"], {}))
        response, attempts, error = transport.api_call(config, key, payload)
        row = {
            "task_id": task["task_id"],
            "model": task["model"],
            "scene_id": task["scene_id"],
            "arm": task["arm"],
            "selected_strategy": task["selected_strategy"],
            "layout": task["layout"],
            "repeat": task["repeat"],
            "layout_seed": task["layout_seed"],
            "prompt_sha256": task["prompt_sha256"],
            "display_to_canonical": task["display_to_canonical"],
            "canonical_truth": task["canonical_truth"],
            "config_sha256": config_hash,
            "api_sampling_seed": None,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "http_attempts": attempts,
            "raw_response": response,
            "ok": False,
            "complete_text_response": False,
            "invalid_reason": error or "",
        }
        if response is not None:
            try:
                choice = response["choices"][0]
                content = choice["message"].get("content")
                row["response_content"] = content
                row["finish_reason"] = choice.get("finish_reason")
                if choice.get("finish_reason") != "stop":
                    raise ValueError("incomplete_or_filtered_response")
                row["complete_text_response"] = True
                parsed, mode = transport.validate_response(
                    content,
                    set(task["display_to_canonical"]),
                    config["reasoning_max_characters"],
                )
                row.update(
                    ok=True,
                    parsed=parsed,
                    parse_mode=mode,
                    invalid_reason="",
                    canonical_choice=task["display_to_canonical"][parsed["choice"]],
                    canonical_suspicious=[
                        task["display_to_canonical"][target]
                        for target in parsed["suspicious_targets"]
                    ],
                )
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                row["invalid_reason"] = str(exc)
        return row

    with raw_path.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=config["max_workers"]) as pool:
            futures = [pool.submit(one, task) for task in pending]
            for future in as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{row['task_id']} ok={row['ok']} reason={row['invalid_reason']}", flush=True)


def scene_bootstrap(reference, candidate, seed=20261119099, repetitions=5000):
    common = [scene for scene in reference if scene in candidate and reference[scene] is not None and candidate[scene] is not None]
    if not common:
        return None
    values = np.array([candidate[scene] - reference[scene] for scene in common])
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "ci_95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "scenes": len(common),
        "p_two_sided_centered": float((1 + np.count_nonzero(np.abs(sampled - values.mean()) >= abs(values.mean()))) / (repetitions + 1)),
    }


def analyze():
    tasks = load(OUTPUT / "tasks.jsonl")
    raw = load(OUTPUT / "raw.jsonl")
    per = {}
    for model in MODELS:
        for arm in ("robust", "SUQR", "RAG_QR"):
            values = {}
            for scene_id in sorted({task["scene_id"] for task in tasks if task["model"] == model}):
                rows = [
                    row for row in raw
                    if row["model"] == model and row["scene_id"] == scene_id and row["arm"] == arm and row["ok"]
                ]
                if rows:
                    utility = []
                    for row in rows:
                        truth = row["canonical_truth"][row["canonical_choice"]]
                        utility.append(3.0 if truth["is_decoy"] else -truth["loss"])
                    values[scene_id] = float(np.mean(utility))
            per[f"{model}|{arm}"] = values
    result = {
        "planned_tasks": len(tasks),
        "completed_tasks": len(raw),
        "valid_tasks": sum(row["ok"] for row in raw),
        "duplicate_task_ids": len(raw) - len({row["task_id"] for row in raw}),
        "models": {},
        "adoption": {},
    }
    positive_suqr = positive_rag = 0
    for model in MODELS:
        robust = per[f"{model}|robust"]
        suqr = per[f"{model}|SUQR"]
        rag = per[f"{model}|RAG_QR"]
        vs_suqr = scene_bootstrap(suqr, robust)
        vs_rag = scene_bootstrap(rag, robust)
        result["models"][model] = {
            "robust_mean_utility": float(np.mean(list(robust.values()))),
            "suqr_mean_utility": float(np.mean(list(suqr.values()))),
            "rag_mean_utility": float(np.mean(list(rag.values()))),
            "vs_suqr": vs_suqr,
            "vs_rag": vs_rag,
        }
        positive_suqr += vs_suqr["ci_95"][0] > 0
        positive_rag += vs_rag["ci_95"][0] > 0
    result["adoption"] = {
        "adopt": positive_suqr >= 2 and positive_rag >= 2,
        "positive_vs_suqr_models": positive_suqr,
        "positive_vs_rag_models": positive_rag,
    }
    (OUTPUT / "summary.json").write_bytes(encoded(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    _, tasks, audit = prepare()
    if args.run:
        run_api(tasks, args.limit)
    if args.analyze:
        print(json.dumps(analyze(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
