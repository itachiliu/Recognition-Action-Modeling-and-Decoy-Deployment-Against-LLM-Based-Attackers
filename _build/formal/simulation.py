"""Simulation under the article's generating equations; never LLM evidence."""
import numpy as np

import design
import estimation as est


def synthetic(seed=20261114122, rho=.5, profile="strong", scenes=None):
    scenes = design.build_scenes() if scenes is None else scenes
    maximum = max(len(s["targets"]) for s in scenes)
    features, labels, choices, masks, ids, splits = [], [], [], [], [], []
    gamma = np.array([1.2, .6, .4, .2, .35, .25, .25, -.1, .15, -.2])
    if profile not in ("strong", "weak"):
        raise ValueError("Unknown recognition profile")
    alpha = np.array([-2, .9, 1.3, 1., 1.1]) if profile == "strong" else np.array([-3, .15, .15, .15, .15])
    rng = np.random.default_rng(seed)
    for scene in scenes:
        item = design.build_presentations([scene])[0]
        parsed = design.base.parse_features(item["prompt"])
        inv = {v: k for k, v in item["display_to_canonical"].items()}
        matrix = np.array([parsed[inv[c]]["x"] + parsed[inv[c]]["z"] for c in sorted(inv)])
        n = len(matrix)
        p = est.sigmoid(np.c_[np.ones(n), matrix[:, 6:]] @ alpha)
        for _ in range(8):
            r = rng.binomial(1, p)
            q = est.probabilities(matrix @ gamma + np.log(rho) * r, np.ones(n, dtype=bool))
            features.append(np.pad(matrix, [(0, maximum - n), (0, 0)]))
            labels.append(np.pad(r, (0, maximum - n)))
            choices.append(rng.choice(n, p=q))
            masks.append(np.arange(maximum) < n)
            ids.append(scene["scene_id"])
            splits.append(scene["split"])
    return {
        "F": np.array(features), "r": np.array(labels, dtype=float), "choice": np.array(choices),
        "mask": np.array(masks), "scene": np.array(ids), "split": np.array(splits),
    }
