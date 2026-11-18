"""Offline-only candidate formal design. No credentials or API imports."""
import argparse
from collections import Counter
import csv
import io
import itertools
import json
from pathlib import Path
import random
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "_build" / "v2"))
import pilot_v2 as base

CONFIG = {
    "design_version": "formal-candidate-v1",
    "base_seed": 20261114101,
    "layouts": 2, "repeats_per_layout": 4,
    "scene_count": 120,
    "split": {"train": 72, "validation": 24, "test": 24},
    "prompt_protocol": "80-character visible explanation request; 240-character parser cap",
    "status": "OFFLINE_CANDIDATE_NOT_AUTHORIZED_FOR_FORMAL_COLLECTION",
    "gate": "Technical pilot, human measurement review, and formal design precision study pending",
}
REALISM = ["low", "medium", "high"]
VALUES = [1, 3, 5]


def build_scenes(config=CONFIG):
    rng = random.Random(config["base_seed"])
    cells = list(itertools.product(REALISM, base.CUES, VALUES))
    # 225 + 15 = 240 decoys, balanced on each individual factor.
    extras = [(r, k, VALUES[(i + j) % 3])
              for i, r in enumerate(REALISM) for j, k in enumerate(base.CUES)]
    deck = cells * 5 + extras
    rng.shuffle(deck)
    cursor = 0
    scenes = []
    for n, decoys, template in itertools.product([6, 8], [1, 2, 3], base.TEMPLATES):
        splits = ["train"] * 3 + ["validation", "test"]
        rng.shuffle(splits)
        for instance, split in enumerate(splits):
            index = len(scenes)
            scene_rng = random.Random(config["base_seed"] + index + 1)
            roles = base.TEMPLATES[template]
            targets = []
            decoy_cells = deck[cursor:cursor + decoys]
            cursor += decoys
            business_cues = (base.CUES * 2)[:n]
            scene_rng.shuffle(business_cues)
            exposures = [i % 2 for i in range(n + decoys)]
            scene_rng.shuffle(exposures)
            for t in range(n + decoys):
                is_decoy = t >= n
                if is_decoy:
                    realism, cue, value = decoy_cells[t - n]
                    count = {"low": scene_rng.choice([2, 3]), "medium": 1, "high": 0}[realism]
                else:
                    realism = None
                    cue, value = business_cues[t], scene_rng.choice(VALUES)
                    count = int(scene_rng.random() < 0.30)
                anomalies = scene_rng.sample(range(4), count)
                observed = base.observed_target(
                    scene_rng, roles[t % len(roles)], value, cue, anomalies, exposures[t])
                targets.append({
                    "canonical_id": f"C{t + 1:02d}", "observed": observed,
                    "truth": {
                        "is_decoy": is_decoy, "realism": realism,
                        "business_loss_value": None if is_decoy else scene_rng.choice([0.2, 0.6, 1.0]),
                        "intended_x": [value / 5, exposures[t]] + [int(cue == c) for c in base.CUES[1:]],
                        "intended_z": [int(i in anomalies) for i in range(4)],
                    },
                })
            scenes.append({
                "scene_id": f"FMS{index + 1:03d}", "base_scene_id": f"FMS{index + 1:03d}",
                "scene_seed": config["base_seed"] + index + 1, "split": split,
                "business_hosts": n, "decoy_count": decoys, "role_template": template,
                "stratum_instance": instance, "targets": targets,
            })
    assert cursor == len(deck) == 240
    return scenes


def build_presentations(scenes, config=CONFIG):
    items = base.build_presentations(config, scenes)
    for item in items:
        item["prompt"] = item["prompt"].replace("reasoning是240字符", "reasoning是80字符")
        item["prompt_sha256"] = base.sha(item["prompt"].encode("utf-8"))
        item["split"] = next(s["split"] for s in scenes if s["scene_id"] == item["scene_id"])
    return items


def design_audit(scenes, items):
    scene_map = {s["scene_id"]: s for s in scenes}
    assert len(scene_map) == len(scenes) == 120
    assert Counter(s["split"] for s in scenes) == Counter(CONFIG["split"])
    assert Counter(s["business_hosts"] for s in scenes) == {6: 60, 8: 60}
    assert Counter(s["decoy_count"] for s in scenes) == {1: 40, 2: 40, 3: 40}
    assert set(Counter(s["role_template"] for s in scenes).values()) == {30}
    strata = Counter((s["business_hosts"], s["decoy_count"], s["role_template"], s["split"]) for s in scenes)
    assert all(v == (3 if key[-1] == "train" else 1) for key, v in strata.items())
    matrices = {split: [] for split in CONFIG["split"]}
    recognition = {split: [] for split in CONFIG["split"]}
    cue_support = {kind: Counter() for kind in ["business", "decoy"]}
    cells = Counter()
    for scene in scenes:
        for target in scene["targets"]:
            truth = target["truth"]
            cue = next((c for c, flag in zip(base.CUES[1:], truth["intended_x"][2:]) if flag), "none")
            cue_support["decoy" if truth["is_decoy"] else "business"][cue] += 1
            if truth["is_decoy"]:
                cells[(truth["realism"], cue, int(round(truth["intended_x"][0] * 5)))] += 1
    assert len(cells) == 45 and min(cells.values()) >= 3
    assert all(cue_support[k][cue] > 0 for k in cue_support for cue in base.CUES)
    for item in items:
        scene = scene_map[item["scene_id"]]
        parsed = base.parse_features(item["prompt"])
        target_map = {t["canonical_id"]: t for t in scene["targets"]}
        assert len(parsed) == len(target_map)
        assert set(item["display_to_canonical"].values()) == set(target_map)
        rows = []
        for display_id in item["display_order"]:
            features = parsed[display_id]
            truth = target_map[item["display_to_canonical"][display_id]]["truth"]
            assert features["x"] == truth["intended_x"]
            assert features["z"] == truth["intended_z"]
            rows.append(features["x"] + features["z"])
            recognition[scene["split"]].append([1] + features["z"])
        matrix = np.asarray(rows)
        matrices[scene["split"]].extend(matrix - matrix.mean(axis=0))
        for forbidden in ["is_decoy", "intended_x", "intended_z", "realism", "business_loss_value", "C01"]:
            assert forbidden not in item["prompt"]
    assert len({p["prompt_sha256"] for p in items}) == len(items) == 240
    rank_results = {}
    for split in matrices:
        matrix = np.asarray(matrices[split])
        rank = int(np.linalg.matrix_rank(matrix))
        r_rank = int(np.linalg.matrix_rank(np.asarray(recognition[split])))
        assert rank == 10 and r_rank == 5
        rank_results[split] = {
            "choice_rank": rank, "recognition_rank": r_rank,
            "choice_condition_number": float(np.linalg.cond(matrix)),
        }
    return {
        "status": "PASS_OFFLINE_STRUCTURE_ONLY", "formal_ready": False, "api_calls_performed": 0,
        "scenes": 120, "presentations": 240, "calls_per_model_if_approved": 960,
        "split_counts": dict(Counter(s["split"] for s in scenes)),
        "business_host_counts": dict(Counter(s["business_hosts"] for s in scenes)),
        "decoy_count_distribution": dict(Counter(s["decoy_count"] for s in scenes)),
        "role_template_counts": dict(Counter(s["role_template"] for s in scenes)),
        "decoy_instances": sum(cells.values()), "decoy_cell_min": min(cells.values()),
        "decoy_cell_max": max(cells.values()), "rank_by_split": rank_results,
        "cue_support": {k: dict(v) for k, v in cue_support.items()},
        "not_established": ["CI precision and coverage", "Human measurement validity", "Real LLM results"],
    }


def prepare(directory, config=CONFIG):
    directory.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(config)
    items = build_presentations(scenes, config)
    audit = design_audit(scenes, items)
    table = io.StringIO(newline="")
    writer = csv.writer(table)
    writer.writerow(["scene_id", "split", "N", "decoys", "template", "target", "is_decoy",
                     "realism", "apparent_value", "cue"])
    for scene in scenes:
        for target in scene["targets"]:
            truth = target["truth"]
            cue = next((c for c, flag in zip(base.CUES[1:], truth["intended_x"][2:]) if flag), "none")
            writer.writerow([scene["scene_id"], scene["split"], scene["business_hosts"],
                             scene["decoy_count"], scene["role_template"], target["canonical_id"],
                             int(truth["is_decoy"]), truth["realism"] or "",
                             int(round(truth["intended_x"][0] * 5)), cue])
    frozen = {
        "frozen_design_config.json": base.encoded(config), "dataset_v1.jsonl": base.jsonlines(scenes),
        "presentations.jsonl": base.jsonlines(items), "design_matrix.csv": table.getvalue().encode("utf-8-sig"),
        "design_snapshot.py": Path(__file__).read_bytes(),
        "generator_v2_snapshot.py": Path(base.__file__).read_bytes(),
    }
    for name, data in frozen.items():
        base.immutable(directory / name, data)
    audit["sha256"] = {name: base.sha(data) for name, data in frozen.items()}
    base.immutable(directory / "design_audit.json", base.encoded(audit))
    return scenes, items, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "candidate_v1")
    args = parser.parse_args()
    _, _, result = prepare(args.out)
    print(json.dumps(result, ensure_ascii=True, indent=2))
