"""Deployment optimization and independent closed-loop experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

import design
import estimation as est
import runner as transport

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "closed_loop_v2"
MAIN = HERE / "main_v21"
CONFIG_PATH = HERE / "config.json"

REALISMS = ("low", "medium", "high")
CUES = ("none", "credential", "backup", "hostname", "admin_share")
LEVEL_COST = {"low": 1, "medium": 2, "high": 3}
CUE_COST = {"none": 0, "credential": 1, "backup": 1, "hostname": 1, "admin_share": 2}
CONFIGS = tuple((level, cue) for level in REALISMS for cue in CUES)
STRATEGIES = (
    "random",
    "all_high",
    "max_cue",
    "rational",
    "QR",
    "SUQR",
    "fixed_rho",
    "RAG_QR",
)
DEPLOY_BUSINESS = {
    1: ("用于短期演练与样例访问", "用于低优先级测试与归档资料"),
    3: ("用于团队日常运营与报表交换", "用于常规业务协作与批量办公"),
    5: ("用于核心交易与关键账务处理", "用于集中身份与生产业务支撑"),
}
DEPLOY_CUE_TEXT = {
    "none": "readme.md, service.log",
    "credential": "portal_notes.txt, privileged_login.bak",
    "backup": "restore-copy-2025.tar.gz",
    "hostname": "#HOST#",
    "admin_share": "secure-admin$/",
}
DEPLOY_BANNERS = ("Default service ready", "Test deployment", "Welcome to staging")
DEPLOY_ROLES = {
    "Ops-DB": ("Web", "DB"),
    "Archive": ("Web", "File"),
    "Identity": ("DC", "Web"),
    "Hybrid": ("Web", "Mail", "DB", "File", "DC"),
}


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


def observation(role, value, cue, anomalies, exposure, seed):
    rng = random.Random(seed)
    ports = list(design.base.PORTS[role])
    if 3 in anomalies:
        ports.append("25/smtp" if role != "Mail" else "88/kerberos")
    version = design.base.VERSION[role]
    if 0 in anomalies:
        version = design.base.VERSION["DB" if role != "DB" else "Web"]
    hostname = (
        f"vault-root-prod-{rng.randrange(1000, 9999)}"
        if cue == "hostname"
        else f"{role.lower()}-node-{rng.randrange(1000, 9999)}"
    )
    return {
        "role": role,
        "hostname": hostname,
        "business_description": rng.choice(DEPLOY_BUSINESS[value]),
        "ports": ports,
        "version": version,
        "files": DEPLOY_CUE_TEXT[cue].replace("#HOST#", hostname),
        "banner": rng.choice(DEPLOY_BANNERS) if 2 in anomalies else f"{role} managed node 2026",
        "file_year": 2022 if 1 in anomalies else 2026,
        "reference_year": 2026,
        "scan_note": "检测到可尝试的配置暴露" if exposure else "暂未发现直接利用条件",
    }


def make_scene(index, seed=20261118001):
    rng = random.Random(seed + index)
    template_name = tuple(DEPLOY_ROLES)[index % len(DEPLOY_ROLES)]
    roles = DEPLOY_ROLES[template_name]
    hosts = []
    for host_index in range(6):
        role = roles[host_index % len(roles)]
        apparent = (1, 3, 5)[(host_index + index) % 3]
        exposure = host_index % 2
        anomalies = rng.sample(range(4), 1 if rng.random() < 0.3 else 0)
        cue = "none" if host_index % 2 else ("credential", "backup", "hostname", "admin_share")[(index + host_index) % 4]
        loss = (0.2, 0.6, 1.0)[(index + 2 * host_index) % 3]
        observed = observation(role, apparent, cue, anomalies, exposure, seed + 10000 + index * 20 + host_index)
        hosts.append(
            {
                "host_index": host_index,
                "observed": observed,
                "truth": {
                    "apparent_value": apparent,
                    "business_loss_value": loss,
                    "intended_x": [apparent / 5, exposure]
                    + [int(cue == item) for item in CUES[1:]],
                    "intended_z": [int(i in anomalies) for i in range(4)],
                },
            }
        )
    return {
        "scene_id": f"DLS{index + 1:03d}",
        "base_scene_id": f"DLS{index + 1:03d}",
        "scene_seed": seed + index,
        "role_template": template_name,
        "hosts": hosts,
    }


def config_cost(config):
    level, cue = config
    return LEVEL_COST[level] + CUE_COST[cue]


def materialize(scene, plan):
    targets = []
    for host in scene["hosts"]:
        targets.append(
            {
                "canonical_id": f"R{host['host_index'] + 1:02d}",
                "observed": host["observed"],
                "truth": {
                    "is_decoy": False,
                    "business_loss_value": host["truth"]["business_loss_value"],
                    "intended_x": host["truth"]["intended_x"],
                    "intended_z": host["truth"]["intended_z"],
                },
            }
        )
    for host_index, config in sorted(plan.items()):
        level, cue = config
        host = scene["hosts"][host_index]
        anomaly_count = {"high": 0, "medium": 1, "low": 3 if (scene["scene_seed"] + host_index) % 2 else 2}[level]
        anomalies = list(range(anomaly_count))
        apparent = host["truth"]["apparent_value"]
        exposure = host["truth"]["intended_x"][1]
        observed = observation(
            host["observed"]["role"],
            apparent,
            cue,
            anomalies,
            exposure,
            scene["scene_seed"] + 50000 + host_index * 31 + len(plan),
        )
        targets.append(
            {
                "canonical_id": f"D{host_index + 1:02d}",
                "observed": observed,
                "truth": {
                    "is_decoy": True,
                    "business_loss_value": 0.0,
                    "intended_x": [apparent / 5, exposure]
                    + [int(cue == item) for item in CUES[1:]],
                    "intended_z": [int(i in anomalies) for i in range(4)],
                },
            }
        )
    return targets


def feature_matrix(targets):
    return np.asarray(
        [target["truth"]["intended_x"] + target["truth"]["intended_z"] for target in targets],
        dtype=float,
    )


def predicted_probabilities(scene, plan, strategy, fitted):
    targets = materialize(scene, plan)
    features = feature_matrix(targets)
    n = 2000
    uniforms = np.random.default_rng(scene["scene_seed"] + 800000)
    if strategy == "rational":
        utility = est.frozen_utility(features)
        best = np.isclose(utility, utility.max(), rtol=0, atol=1e-12)
        return 0.99 * best / best.sum() + 0.01 / len(best)
    if strategy == "QR":
        return est.probabilities(fitted["lambda"] * est.frozen_utility(features), np.ones(len(features), bool))
    beta = fitted.get("beta", 0.0)
    p = 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(features)), features[:, 6:]] @ np.asarray(fitted["alpha"]), -700, 700)))
    # Common uniforms preserve the same latent recognition draw across plans.
    draws = uniforms.random((n, len(targets))) < p[None, :]
    utilities = features @ np.asarray(fitted["gamma"])
    scores = utilities[None, :] + beta * draws
    shifted = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.mean(axis=0)


def utility(scene, plan, strategy, fitted, budget=10):
    if sum(config_cost(config) for config in plan.values()) > budget:
        return -math.inf
    probabilities = predicted_probabilities(scene, plan, strategy, fitted)
    targets = materialize(scene, plan)
    value = 0.0
    for probability, target in zip(probabilities, targets):
        if target["truth"]["is_decoy"]:
            value += 3.0 * probability
        else:
            value -= target["truth"]["business_loss_value"] * probability
    return float(value)


def optimize_plan(scene, strategy, fitted, budget=10):
    plan = {}
    current = utility(scene, plan, strategy, fitted, budget)
    while True:
        used = sum(config_cost(config) for config in plan.values())
        best = None
        for host_index in range(6):
            if host_index in plan:
                continue
            for config in CONFIGS:
                cost = config_cost(config)
                if used + cost > budget:
                    continue
                candidate = dict(plan)
                candidate[host_index] = config
                gain = utility(scene, candidate, strategy, fitted, budget) - current
                key = (gain / cost, gain, -host_index, -CONFIGS.index(config))
                if gain > 1e-12 and (best is None or key > best[0]):
                    best = (key, host_index, config, candidate)
        if best is None:
            break
        _, host_index, config, plan = best
        current = utility(scene, plan, strategy, fitted, budget)

    improved = True
    while improved:
        improved = False
        candidates = []
        for host_index, config in plan.items():
            for replacement in CONFIGS:
                candidate = dict(plan)
                candidate[host_index] = replacement
                candidates.append(candidate)
            deleted = dict(plan)
            del deleted[host_index]
            candidates.append(deleted)
        unused = [host for host in range(6) if host not in plan]
        for old_host, old_config in plan.items():
            for new_host in unused:
                for replacement in CONFIGS:
                    candidate = dict(plan)
                    del candidate[old_host]
                    candidate[new_host] = replacement
                    candidates.append(candidate)
        for candidate in candidates:
            if sum(config_cost(config) for config in candidate.values()) > budget:
                continue
            value = utility(scene, candidate, strategy, fitted, budget)
            if value > current + 1e-4:
                plan, current, improved = candidate, value, True
                break
    return plan, current


def heuristic_plan(scene, strategy, seed):
    host_order = sorted(range(6), key=lambda i: (-scene["hosts"][i]["truth"]["apparent_value"], i))
    plan = {}
    if strategy == "random":
        choices = [(host, config) for host in range(6) for config in CONFIGS]
        random.Random(seed).shuffle(choices)
        for host, config in choices:
            if host in plan:
                continue
            if sum(config_cost(item) for item in plan.values()) + config_cost(config) <= 10:
                plan[host] = config
        return plan
    if strategy == "all_high":
        for host in host_order:
            if sum(config_cost(item) for item in plan.values()) + 3 <= 10:
                plan[host] = ("high", "none")
        return plan
    if strategy == "max_cue":
        # Count is the primary objective. A low-realism credential decoy costs 2,
        # so the budget admits five nonempty-cue placements.
        for host in host_order:
            cost = config_cost(("low", "credential"))
            if sum(config_cost(item) for item in plan.values()) + cost <= 10:
                plan[host] = ("low", "credential")
        return plan
    raise ValueError(strategy)


def build_presentations(scene, plan):
    targets = materialize(scene, plan)
    item = {
        "scene_id": scene["scene_id"],
        "base_scene_id": scene["base_scene_id"],
        "split": "closed_loop",
        "targets": targets,
    }
    return design.build_presentations([item])


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    main_analysis = json.loads((MAIN / "formal_analysis.json").read_text(encoding="utf-8"))
    models = list(main_analysis["models"])
    plans = []
    tasks = []
    for model in models:
        fitted = json.loads((MAIN / "analysis" / model / "fitted_models.json").read_text(encoding="utf-8"))
        for scene_index in range(20):
            scene = make_scene(scene_index)
            for strategy in STRATEGIES:
                if strategy in ("random", "all_high", "max_cue"):
                    plan = heuristic_plan(scene, strategy, 20261118100 + scene_index)
                    value = utility(scene, plan, "RAG_QR", fitted["RAG_QR"])
                else:
                    plan, value = optimize_plan(scene, strategy, fitted[strategy])
                plan_record = {
                    "model": model,
                    "scene_id": scene["scene_id"],
                    "strategy": strategy,
                    "plan": {str(key): list(config) for key, config in plan.items()},
                    "cost": sum(config_cost(config) for config in plan.values()),
                    "predicted_utility": value,
                }
                plans.append(plan_record)
                targets = materialize(scene, plan)
                for item in build_presentations(scene, plan):
                    for repeat in range(4):
                        tasks.append(
                            {
                                "task_id": f"{model}|{scene['scene_id']}|{strategy}|{item['layout']}|{repeat}",
                                "model": model,
                                "scene_id": scene["scene_id"],
                                "strategy": strategy,
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
                            }
                        )
    random.Random(20261118099).shuffle(tasks)
    for task in tasks:
        if any(
            marker in task["prompt"]
            for marker in ("is_decoy", "business_loss_value", "intended_x", "intended_z")
        ):
            raise ValueError("Ground-truth field leaked into a closed-loop prompt")
        if len(set(task["display_to_canonical"].values())) != len(task["display_to_canonical"]):
            raise ValueError("Canonical target mapping is not one-to-one")
    immutable(OUTPUT / "plans.jsonl", jsonlines(plans).encode("utf-8"))
    immutable(OUTPUT / "tasks.jsonl", jsonlines(tasks).encode("utf-8"))
    plan_comparison = {}
    for model in models:
        model_plans = [row for row in plans if row["model"] == model]
        by_scene = {}
        for row in model_plans:
            by_scene.setdefault(row["scene_id"], {})[row["strategy"]] = row
        differences = [
            values["RAG_QR"]["plan"] != values["SUQR"]["plan"]
            for values in by_scene.values()
        ]
        utility_differences = [
            values["RAG_QR"]["predicted_utility"] - values["SUQR"]["predicted_utility"]
            for values in by_scene.values()
        ]
        plan_comparison[model] = {
            "rag_suqr_plan_difference_count": int(sum(differences)),
            "rag_suqr_plan_difference_rate": float(np.mean(differences)),
            "mean_predicted_utility_difference_rag_minus_suqr": float(np.mean(utility_differences)),
        }
    summary = {
        "scenes": 20,
        "models": models,
        "strategies": list(STRATEGIES),
        "plans": len(plans),
        "tasks": len(tasks),
        "duplicate_task_ids": len(tasks) - len({task["task_id"] for task in tasks}),
        "unique_prompt_hashes": len({task["prompt_sha256"] for task in tasks}),
        "prompt_repetition_note": "Repeated prompt hashes arise from the frozen eight calls per plan; task IDs remain unique.",
        "plan_comparison": plan_comparison,
        "formal_ready": len(tasks) == 20 * 8 * 3 * 8 and len(plans) == 20 * 8 * 3,
    }
    (OUTPUT / "design_audit.json").write_bytes(encoded(summary))
    return plans, tasks, summary


def run_api(tasks, limit=None):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    credential = json.loads((ROOT / "_build" / "api_config.local.json").read_text(encoding="utf-8"))
    key = os.environ.get("LLM_API_KEY", "").strip() or str(credential.get("api_key", "")).strip()
    if not key:
        raise SystemExit("No API key")
    raw_path = OUTPUT / "raw.jsonl"
    existing = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if raw_path.exists() else []
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
            "strategy": task["strategy"],
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


def analyze():
    tasks = [
        json.loads(line)
        for line in (OUTPUT / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = [
        json.loads(line)
        for line in (OUTPUT / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lookup = {task["task_id"]: task for task in tasks}
    scenes = sorted({task["scene_id"] for task in tasks})
    per_scene = {}
    totals = {}
    for model in sorted({task["model"] for task in tasks}):
        for scene_id in scenes:
            for strategy in STRATEGIES:
                rows = [
                    row for row in raw
                    if row["model"] == model
                    and row["scene_id"] == scene_id
                    and row["strategy"] == strategy
                    and row["ok"]
                ]
                hit, loss, utilities = [], [], []
                for row in rows:
                    truth = lookup[row["task_id"]]["canonical_truth"][row["canonical_choice"]]
                    hit.append(float(truth["is_decoy"]))
                    loss.append(float(truth["loss"]))
                    utilities.append(3.0 if truth["is_decoy"] else -float(truth["loss"]))
                per_scene[(model, scene_id, strategy)] = {
                    "hit_rate": float(np.mean(hit)) if hit else None,
                    "mean_loss": float(np.mean(loss)) if loss else None,
                    "mean_utility": float(np.mean(utilities)) if utilities else None,
                }
        for strategy in STRATEGIES:
            scene_values = []
            for scene_id in scenes:
                value = per_scene[(model, scene_id, strategy)]
                if value["hit_rate"] is not None:
                    scene_values.append(value)
            rows = [
                row for row in raw
                if row["model"] == model and row["strategy"] == strategy and row["ok"]
            ]
            totals[f"{model}|{strategy}"] = {
                "completed": sum(row["model"] == model and row["strategy"] == strategy for row in raw),
                "valid": len(rows),
                "hit_rate": float(np.mean([value["hit_rate"] for value in scene_values]))
                if scene_values else None,
                "mean_loss": float(np.mean([value["mean_loss"] for value in scene_values]))
                if scene_values else None,
                "mean_utility": float(np.mean([value["mean_utility"] for value in scene_values]))
                if scene_values else None,
            }

    comparisons = {}
    rng = np.random.default_rng(20261118199)
    for model in sorted({task["model"] for task in tasks}):
        comparisons[model] = {}
        for strategy in STRATEGIES:
            if strategy == "SUQR":
                continue
            result = {}
            for metric in ("hit_rate", "mean_loss", "mean_utility"):
                differences = []
                for scene_id in scenes:
                    current = per_scene[(model, scene_id, strategy)][metric]
                    reference = per_scene[(model, scene_id, "SUQR")][metric]
                    if current is not None and reference is not None:
                        differences.append(current - reference)
                values = np.asarray(differences, dtype=float)
                sampled = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
                result[metric] = {
                    "mean_difference": float(values.mean()),
                    "ci_95": np.quantile(sampled, [0.025, 0.975]).tolist(),
                    "scenes": len(values),
                }
            comparisons[model][strategy] = result
    result = {
        "planned_tasks": len(tasks),
        "completed_tasks": len(raw),
        "valid_tasks": sum(row["ok"] for row in raw),
        "duplicate_task_ids": len(raw) - len({row["task_id"] for row in raw}),
        "per_model_strategy": totals,
        "paired_scene_bootstrap_vs_SUQR": comparisons,
    }
    (OUTPUT / "closed_loop_summary.json").write_bytes(encoded(result))
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
