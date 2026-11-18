"""Formal-v2.1 collector; default action is offline freezing and audit only."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_build" / "formal"))
sys.path.insert(0, str(ROOT / "_build" / "v25"))

import design
import expanded_design_diagnostic as expanded
import runner as transport


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen file differs: {path}. Use a new protocol version.")
    else:
        path.write_bytes(data)


def jsonlines(items):
    return "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items
    ).encode("utf-8")


def prepare(config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    scenes = expanded.build_scenes()
    presentations, design_audit = expanded.audit(scenes)
    expected = len(presentations) * config["repeats_per_layout"] * len(config["models"])
    if len(scenes) != config["scene_count"]:
        raise ValueError("Frozen scene count differs from protocol")
    if expected != 9216:
        raise ValueError(f"Unexpected formal task count: {expected}")
    config_data = encoded(config)
    immutable(directory / "frozen_config.json", config_data)
    immutable(directory / "dataset_v2.jsonl", jsonlines(scenes))
    immutable(directory / "presentations.jsonl", jsonlines(presentations))
    features = [
        {
            "scene_id": item["scene_id"],
            "layout": item["layout"],
            "features": design.base.parse_features(item["prompt"]),
        }
        for item in presentations
    ]
    immutable(directory / "features_from_text.jsonl", jsonlines(features))
    immutable(directory / "design_audit_v2.json", encoded(design_audit))
    immutable(directory / "runner_snapshot.py", Path(__file__).read_bytes())
    immutable(directory / "transport_snapshot.py", Path(transport.__file__).read_bytes())
    immutable(directory / "estimation_snapshot.py", (ROOT / "_build" / "formal" / "estimation.py").read_bytes())
    audit = {
        "protocol_version": config["protocol_version"],
        "formal_ready": True,
        "api_calls_performed": 0,
        "scenes": len(scenes),
        "presentations": len(presentations),
        "models": config["models"],
        "planned_tasks": expected,
        "duplicate_prompt_hashes": len(presentations)
        - len({item["prompt_sha256"] for item in presentations}),
        "design_audit": design_audit,
        "sha256": {
            name: sha((directory / name).read_bytes())
            for name in [
                "frozen_config.json",
                "dataset_v2.jsonl",
                "presentations.jsonl",
                "features_from_text.jsonl",
                "design_audit_v2.json",
                "runner_snapshot.py",
                "transport_snapshot.py",
                "estimation_snapshot.py",
            ]
        },
        "scope": "Frozen formal protocol; offline audit only until explicit API run",
    }
    immutable(directory / "offline_audit.json", encoded(audit))
    return presentations, audit


def summarize(config, directory):
    rows = [
        json.loads(line)
        for line in (directory / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if (directory / "raw.jsonl").exists() else []
    expected = config["scene_count"] * 8
    result = {
        "protocol_version": config["protocol_version"],
        "planned_per_model": expected,
        "completed_tasks": len(rows),
        "duplicate_task_ids": len(rows) - len({row["task_id"] for row in rows}),
        "per_model": {},
    }
    for model in config["models"]:
        subset = [row for row in rows if row["model"] == model]
        complete = [row for row in subset if row.get("complete_text_response")]
        valid = [row for row in subset if row.get("ok")]
        result["per_model"][model] = {
            "completed": len(subset),
            "complete_text": len(complete),
            "valid": len(valid),
            "complete_given_http200": len(complete) / len(subset) if subset else None,
            "valid_given_complete": len(valid) / len(complete) if complete else None,
        }
    (directory / "run_summary.json").write_bytes(encoded(result))
    return result


def run(config, presentations, directory, api_config, limit):
    credential = json.loads(api_config.read_text(encoding="utf-8")) if api_config.exists() else {}
    key = os.environ.get("LLM_API_KEY", "").strip() or str(credential.get("api_key", "")).strip()
    if not key or key == "在这里填入你的 API Key":
        raise SystemExit("Configure LLM_API_KEY or the existing local credential file.")
    if credential.get("endpoint") not in (None, "", config["endpoint"]):
        raise ValueError("Credential endpoint differs from frozen protocol")
    raw = directory / "raw.jsonl"
    existing = [
        json.loads(line)
        for line in raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if raw.exists() else []
    completed = {row["task_id"] for row in existing}
    config_hash = sha(encoded(config))
    if any(row["config_sha256"] != config_hash for row in existing):
        raise ValueError("Existing results belong to another protocol")
    jobs = []
    for model in config["models"]:
        for item in presentations:
            for repeat in range(config["repeats_per_layout"]):
                task_id = f"{model}|{item['scene_id']}|{item['layout']}|{repeat}"
                if task_id not in completed:
                    jobs.append((task_id, model, item, repeat))
    random.Random(config["base_seed"] + 900000).shuffle(jobs)
    if limit is not None:
        jobs = jobs[:limit]

    def one(job):
        task_id, model, item, repeat = job
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "max_tokens": config["max_tokens_by_model"][model],
        }
        if model in config["disable_thinking_request_models"]:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.update(config.get("request_overrides", {}).get(model, {}))
        started = datetime.now(timezone.utc).isoformat()
        response, attempts, error = transport.api_call(config, key, payload)
        row = {
            "task_id": task_id,
            "model": model,
            "scene_id": item["scene_id"],
            "base_scene_id": item["base_scene_id"],
            "layout": item["layout"],
            "repeat": repeat,
            "layout_seed": item["layout_seed"],
            "api_sampling_seed": None,
            "config_sha256": config_hash,
            "started_utc": started,
            "prompt_sha256": item["prompt_sha256"],
            "display_to_canonical": item["display_to_canonical"],
            "request": payload,
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
                if not isinstance(content, str):
                    raise ValueError("non_text_content")
                row["complete_text_response"] = True
                parsed, mode = transport.validate_response(
                    content,
                    set(item["display_to_canonical"]),
                    config["reasoning_max_characters"],
                )
                row.update(
                    ok=True,
                    parsed=parsed,
                    parse_mode=mode,
                    invalid_reason="",
                    canonical_choice=item["display_to_canonical"][parsed["choice"]],
                    canonical_suspicious=[
                        item["display_to_canonical"][target]
                        for target in parsed["suspicious_targets"]
                    ],
                )
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                row["invalid_reason"] = str(exc)
        return row

    with raw.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=config["max_workers"]) as pool:
            futures = [pool.submit(one, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{row['task_id']} ok={row['ok']} reason={row['invalid_reason']}", flush=True)
    return summarize(config, directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--out", type=Path, default=HERE / "main_v21")
    parser.add_argument("--api-config", type=Path, default=ROOT / "_build" / "api_config.local.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    presentations, audit = prepare(config, args.out)
    if args.run:
        print(json.dumps(run(config, presentations, args.out, args.api_config, args.limit), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print("OFFLINE ONLY. No credentials loaded; no API calls made.")


if __name__ == "__main__":
    main()
