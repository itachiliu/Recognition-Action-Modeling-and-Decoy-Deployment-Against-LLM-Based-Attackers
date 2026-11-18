"""Final technical audit for formal-v2.1."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "main_v21"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def load(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    config = json.loads((RUN / "frozen_config.json").read_text(encoding="utf-8"))
    presentations = load(RUN / "presentations.jsonl")
    records = load(RUN / "raw.jsonl")
    expected = len(presentations) * config["repeats_per_layout"] * len(config["models"])
    config_hash = hashlib.sha256(encoded(config)).hexdigest()
    expected_ids = {
        f"{model}|{item['scene_id']}|{item['layout']}|{repeat}"
        for model in config["models"]
        for item in presentations
        for repeat in range(config["repeats_per_layout"])
    }
    actual_ids = [record["task_id"] for record in records]
    per_model = {}
    for model in config["models"]:
        rows = [record for record in records if record["model"] == model]
        complete = [record for record in rows if record.get("complete_text_response")]
        valid = [record for record in rows if record.get("ok")]
        per_model[model] = {
            "planned": config["scene_count"] * 8,
            "completed": len(rows),
            "complete_text": len(complete),
            "valid": len(valid),
            "complete_given_completed": len(complete) / len(rows) if rows else None,
            "valid_given_complete": len(valid) / len(complete) if complete else None,
            "finish_reasons": dict(Counter(str(record.get("finish_reason")) for record in rows)),
            "errors": dict(
                Counter(record.get("invalid_reason") for record in rows if not record.get("ok"))
            ),
            "reported_response_model_ids": dict(
                Counter(str((record.get("raw_response") or {}).get("model")) for record in rows)
            ),
        }
    result = {
        "protocol_version": config["protocol_version"],
        "planned_tasks": expected,
        "completed_tasks": len(records),
        "unique_task_ids": len(set(actual_ids)),
        "duplicate_task_ids": len(actual_ids) - len(set(actual_ids)),
        "missing_expected_task_ids": len(expected_ids - set(actual_ids)),
        "unexpected_task_ids": len(set(actual_ids) - expected_ids),
        "config_sha256_expected": config_hash,
        "wrong_config_hash": sum(record.get("config_sha256") != config_hash for record in records),
        "per_model": per_model,
        "formal_collection_complete": (
            len(records) == expected
            and len(set(actual_ids)) == expected
            and expected_ids == set(actual_ids)
            and all(record.get("config_sha256") == config_hash for record in records)
        ),
    }
    (RUN / "formal_technical_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
