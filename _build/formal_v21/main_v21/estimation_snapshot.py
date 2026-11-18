"""Two-stage RAG-QR, held-out marginal prediction and scene bootstrap.

This module is offline. Labels can enter training likelihoods only; marginal
prediction intentionally has no recognition-label argument.
"""
import itertools
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2"))
import pilot_v2 as protocol
from check_model import newton, sigmoid

RIDGES = (1e-4, 1e-3, 1e-2, 1e-1)
MODELS = ("rational", "QR", "SUQR", "fixed_rho", "RAG_QR")
BETA_BOUNDS = (-30.0, 0.0)


def probabilities(scores, mask):
    masked = np.where(mask, scores, -np.inf)
    maximum = np.max(masked, axis=-1, keepdims=True)
    weights = np.where(mask, np.exp(masked - maximum), 0)
    return weights / weights.sum(axis=-1, keepdims=True)


def logistic_objective(theta, z, labels, weights, ridge):
    total = weights.sum()
    if total <= 0:
        raise ValueError("No recognition observations")
    scores = z @ theta
    probs = sigmoid(scores)
    penalty = np.r_[0., np.ones(len(theta) - 1)]
    return (
        float(np.dot(weights, np.logaddexp(0, scores) - labels * scores) / total
              + ridge * np.sum(penalty * theta ** 2) / 2),
        z.T @ (weights * (probs - labels)) / total + ridge * penalty * theta,
        z.T @ ((weights * probs * (1 - probs))[:, None] * z) / total + ridge * np.diag(penalty),
    )


def choice_objective(theta, features, choices, mask, weights, ridge, free_beta, offset=None):
    total = weights.sum()
    if total <= 0:
        raise ValueError("No choices")
    scores = features @ theta
    if offset is not None:
        scores = scores + offset
    probs = probabilities(scores, mask)
    selected = scores[np.arange(len(choices)), choices]
    loss = np.logaddexp.reduce(np.where(mask, scores, -np.inf), axis=1) - selected
    expected = np.einsum("nt,ntd->nd", probs, features)
    diff = expected - features[np.arange(len(choices)), choices]
    hessian = (
        np.einsum("n,nt,ntd,nte->de", weights, probs, features, features, optimize=True)
        - expected.T @ (weights[:, None] * expected)
    ) / total
    penalty = np.ones(len(theta))
    if free_beta:
        penalty[-1] = 0
    return (
        float(weights @ loss / total + ridge * np.sum(penalty * theta ** 2) / 2),
        weights @ diff / total + ridge * penalty * theta,
        hessian + ridge * np.diag(penalty),
    )


def dataset_from_records(presentations, records, split_by_scene):
    """Only visible prompt features, ID maps and valid responses enter the fit."""
    lookup = {(p["scene_id"], p["layout"]): p for p in presentations}
    rows, seen = [], set()
    maximum = max(len(p["display_to_canonical"]) for p in presentations)
    for record in records:
        if record["task_id"] in seen:
            raise ValueError("Duplicate task ID")
        seen.add(record["task_id"])
        if not record["ok"]:
            continue
        item = lookup[(record["scene_id"], record["layout"])]
        if record["prompt_sha256"] != item["prompt_sha256"]:
            raise ValueError("Response and prompt hash differ")
        canonical = sorted(item["display_to_canonical"].values())
        inverse = {v: k for k, v in item["display_to_canonical"].items()}
        features = protocol.parse_features(item["prompt"])
        matrix = np.zeros((maximum, 10))
        recognition = np.zeros(maximum)
        for i, key in enumerate(canonical):
            visible = features[inverse[key]]
            matrix[i] = visible["x"] + visible["z"]
            recognition[i] = key in record["canonical_suspicious"]
        rows.append({
            "F": matrix, "r": recognition, "choice": canonical.index(record["canonical_choice"]),
            "mask": np.arange(maximum) < len(canonical), "scene": item["scene_id"],
            "split": split_by_scene[item["scene_id"]], "task": record["task_id"],
        })
    if not rows:
        raise ValueError("No valid observations")
    return {key: np.asarray([r[key] for r in rows]) for key in rows[0]}


def subset(data, selector):
    return {key: value[selector] for key, value in data.items()}


def recognition_design(data):
    return np.concatenate([np.ones(data["mask"].shape + (1,)), data["F"][..., 6:]], axis=-1)


def fit_recognition(data, ridge, weights):
    mask = data["mask"]
    z = recognition_design(data)[mask]
    labels = data["r"][mask]
    target_weights = np.broadcast_to(weights[:, None], mask.shape)[mask]
    active = labels[target_weights > 0]
    if len(active) == 0 or np.all(active == active[0]):
        raise ValueError("Recognition intercept has no finite estimate: only one observed label class")
    theta, status = newton(lambda t: logistic_objective(t, z, labels, target_weights, ridge), 5)
    return theta, status


def fit_choice(data, ridge, weights, model, beta_bounds=BETA_BOUNDS):
    free_beta = model == "RAG_QR"
    features = (np.concatenate([data["F"], data["r"][..., None]], axis=-1)
                if free_beta else data["F"])
    if free_beta:
        centered = []
        for f, mask, weight in zip(features, data["mask"], weights):
            if weight > 0:
                visible = f[mask]
                centered.extend(visible - visible.mean(axis=0))
        matrix = np.asarray(centered)
        if matrix.ndim != 2 or np.linalg.matrix_rank(matrix) < 11:
            raise ValueError("Conditional rho not identified by within-choice design; report unavailable, do not impute rho")
    beta = None if free_beta else math.log(0.01) if model == "fixed_rho" else 0.
    offset = None if free_beta else beta * data["r"]
    theta, status = newton(
        lambda t: choice_objective(t, features, data["choice"], data["mask"], weights,
                                   ridge, free_beta, offset),
        features.shape[-1], beta_bounds=beta_bounds if free_beta else None)
    return (theta[:-1], float(theta[-1]), status) if free_beta else (theta, beta, status)


def frozen_utility(features):
    return 2 * features[..., 0] + features[..., 1] + .5 * features[..., 2:6].sum(axis=-1) - .5 * features[..., 6:].sum(axis=-1)


def exact_marginal(features, alpha, gamma, beta):
    """Equation (4). No observed r or ground-truth labels can be supplied."""
    if beta == 0:
        return probabilities(features @ gamma, np.ones(len(features), dtype=bool))
    count = len(features)
    bits = np.asarray(list(itertools.product([0., 1.], repeat=count)))
    p = sigmoid(np.c_[np.ones(count), features[:, 6:]] @ alpha)
    mass = np.prod(np.where(bits, p, 1 - p), axis=1)
    scores = (features @ gamma)[None, :] + beta * bits
    conditional = probabilities(scores, np.ones_like(scores, dtype=bool))
    return mass @ conditional


def predict(data, fitted):
    result = np.zeros_like(data["mask"], dtype=float)
    model = fitted["model"]
    # Canonical ordering makes all eight presentations of a scene identical.
    for scene in np.unique(data["scene"]):
        indices = np.flatnonzero(data["scene"] == scene)
        first = indices[0]
        mask = data["mask"][first]
        features = data["F"][first][mask]
        if not np.all(data["F"][indices] == data["F"][first]):
            raise ValueError("Scene feature mismatch across repeats")
        utility = frozen_utility(features)
        if model == "rational":
            best = np.isclose(utility, utility.max(), rtol=0, atol=1e-12)
            q = .99 * best / best.sum() + .01 / len(best)
        elif model == "QR":
            q = probabilities(fitted["lambda"] * utility, np.ones(len(utility), dtype=bool))
        else:
            q = exact_marginal(features, np.asarray(fitted["alpha"]), np.asarray(fitted["gamma"]), fitted["beta"])
        result[np.ix_(indices, np.flatnonzero(mask))] = q
    return result


def marginal_nll(data, fitted, weights=None):
    weights = np.ones(len(data["choice"])) if weights is None else weights
    q = predict(data, fitted)
    return float(np.dot(weights, -np.log(np.maximum(q[np.arange(len(q)), data["choice"]], 1e-300))) / weights.sum())


def recognition_nll(data, alpha, weights):
    z = recognition_design(data)
    scores = z @ alpha
    loss = np.logaddexp(0, scores) - data["r"] * scores
    w = data["mask"] * weights[:, None]
    return float((loss * w).sum() / w.sum())


def fit_models(train, validation, train_weights=None, validation_weights=None,
               models=MODELS, ridges=RIDGES, beta_bounds=BETA_BOUNDS):
    tw = np.ones(len(train["choice"])) if train_weights is None else train_weights
    vw = np.ones(len(validation["choice"])) if validation_weights is None else validation_weights
    recognition_candidates = []
    for ridge in ridges:
        alpha, status = fit_recognition(train, ridge, tw)
        recognition_candidates.append((recognition_nll(validation, alpha, vw), ridge, alpha, status))
    _, recognition_ridge, alpha, recognition_status = min(recognition_candidates, key=lambda v: v[:2])
    output = {}
    for model in models:
        fitted = {"model": model}
        if model == "rational":
            output[model] = fitted
            continue
        if model == "QR":
            lambdas = (0, .1, .25, .5, 1, 2, 4, 8, 16)
            fitted["lambda"] = min(lambdas, key=lambda value: marginal_nll(
                validation, {"model": "QR", "lambda": value}, vw))
            output[model] = fitted
            continue
        candidates = []
        for ridge in ridges:
            gamma, beta, status = fit_choice(train, ridge, tw, model, beta_bounds)
            candidate = {
                "model": model, "alpha": alpha.tolist(), "gamma": gamma.tolist(),
                "beta": beta, "rho": math.exp(beta), "choice_ridge": ridge,
                "recognition_ridge": recognition_ridge, "choice_fit": status,
                "recognition_fit": recognition_status,
                "beta_at_upper_bound": abs(beta) < 1e-8,
                "beta_at_lower_bound": beta <= beta_bounds[0] + 1e-8,
                "validation_selection": "recognition Bernoulli NLL; choice independent marginal NLL",
            }
            candidates.append((marginal_nll(validation, candidate, vw), ridge, candidate))
        output[model] = min(candidates, key=lambda v: v[:2])[2]
    return output


def prediction_metrics(data, fitted):
    q = predict(data, fitted)
    indices = np.arange(len(q))
    selected = np.maximum(q[indices, data["choice"]], 1e-300)
    top = q.argmax(axis=1)
    confidence = q[indices, top]
    correct = (top == data["choice"]).astype(float)
    # Multiclass Brier sum, padding contributes zero.
    brier = (q * q).sum(axis=1) - 2 * q[indices, data["choice"]] + 1
    ece = 0.
    for b in range(10):
        members = (confidence >= b / 10) & ((confidence < (b + 1) / 10) if b < 9 else (confidence <= 1))
        if members.any():
            ece += members.mean() * abs(correct[members].mean() - confidence[members].mean())
    per_scene = {}
    for scene in np.unique(data["scene"]):
        rows = data["scene"] == scene
        per_scene[str(scene)] = {
            "nll": float((-np.log(selected[rows])).mean()),
            "top1": float(correct[rows].mean()), "brier": float(brier[rows].mean()),
            "valid_tasks": int(rows.sum()),
        }
    return {"nll": float(-np.log(selected).mean()), "top1": float(correct.mean()),
            "brier": float(brier.mean()), "top_label_ece_10_bins": float(ece),
            "per_scene": per_scene}


def bootstrap_weights(data, rng):
    scenes = np.unique(data["scene"])
    counts = rng.multinomial(len(scenes), np.full(len(scenes), 1 / len(scenes)))
    index = {scene: i for i, scene in enumerate(scenes)}
    return np.array([counts[index[scene]] for scene in data["scene"]], dtype=float)


def bootstrap_iteration(train, validation, seed, beta_bounds=BETA_BOUNDS):
    rng = np.random.default_rng(seed)
    # Retune both ridges using scene resamples of their own original split.
    fitted = fit_models(train, validation, bootstrap_weights(train, rng),
                        bootstrap_weights(validation, rng), models=("RAG_QR",),
                        beta_bounds=beta_bounds)["RAG_QR"]
    return {
        "seed": int(seed), "rho": fitted["rho"], "beta": fitted["beta"],
        "alpha": fitted["alpha"], "gamma": fitted["gamma"],
        "choice_converged": fitted["choice_fit"]["converged"],
        "recognition_converged": fitted["recognition_fit"]["converged"],
        "upper_boundary": fitted["beta_at_upper_bound"],
        "lower_boundary": fitted["beta_at_lower_bound"],
        "choice_ridge": fitted["choice_ridge"], "recognition_ridge": fitted["recognition_ridge"],
    }


def summarize_bootstrap(rows, planned=2000):
    valid = [r for r in rows if r.get("choice_converged") and r.get("recognition_converged")]
    values = np.asarray([r["rho"] for r in valid])
    ci = np.quantile(values, [.025, .975]).tolist() if len(values) else None
    return {
        "planned": planned, "completed": len(rows), "successful": len(valid),
        "failure_rate": 1 - len(valid) / len(rows) if rows else None,
        "upper_boundary_count": sum(r.get("upper_boundary", False) for r in rows),
        "lower_boundary_count": sum(r.get("lower_boundary", False) for r in rows),
        "rho_percentile_ci_95": ci,
        "alpha_percentile_ci_95": np.quantile([r["alpha"] for r in valid], [.025, .975], axis=0).tolist() if valid else None,
        "gamma_percentile_ci_95": np.quantile([r["gamma"] for r in valid], [.025, .975], axis=0).tolist() if valid else None,
        "full_2000_complete": len(rows) >= planned == 2000,
        "interval_scope": "Scene bootstrap, training and validation split resampled separately, ridges reselected; failed fits excluded and counted",
    }


def design_diagnostics(data):
    features_r = np.concatenate([data["F"], data["r"][..., None]], axis=-1)
    centered = []
    constant_recognition_calls = 0
    for features, mask, r in zip(features_r, data["mask"], data["r"]):
        visible = features[mask]
        centered.extend(visible - visible.mean(axis=0))
        constant_recognition_calls += int(np.all(r[mask] == r[mask][0]))
    matrix = np.asarray(centered)
    recognition = data["r"][data["mask"]]
    chosen_r = data["r"][np.arange(len(data["choice"])), data["choice"]]
    z = recognition_design(data)[data["mask"]]
    return {
        "valid_choices": len(data["choice"]), "scenes": len(np.unique(data["scene"])),
        "recognition_positive": int(recognition.sum()), "recognition_pairs": len(recognition),
        "chosen_marked": int(chosen_r.sum()),
        "constant_recognition_within_choice_calls": constant_recognition_calls,
        "centered_augmented_choice_rank": int(np.linalg.matrix_rank(matrix)),
        "centered_augmented_condition_number": float(np.linalg.cond(matrix)),
        "recognition_feature_rank": int(np.linalg.matrix_rank(z)),
        "warnings": [
            text for condition, text in [
                (len(np.unique(recognition)) < 2, "No finite unpenalized recognition intercept"),
                (np.linalg.matrix_rank(matrix) < 11, "rho not identified by augmented within-choice design"),
                (chosen_r.sum() == 0, "No marked target chosen: inspect separation/profile; do not conclude rho=0"),
            ] if condition
        ],
    }


def paired_scene_comparison(metrics_a, metrics_b, seed=20261114190, repetitions=2000):
    common = sorted(set(metrics_a["per_scene"]) & set(metrics_b["per_scene"]))
    if not common:
        raise ValueError("No common scenes")
    # Scene means are paired; report this estimand explicitly with missing data.
    differences = np.array([metrics_a["per_scene"][s]["nll"] - metrics_b["per_scene"][s]["nll"] for s in common])
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(0, len(common), size=(repetitions, len(common)))].mean(axis=1)
    centered = sampled - differences.mean()
    p = min(1., (1 + np.count_nonzero(np.abs(centered) >= abs(differences.mean()))) / (repetitions + 1))
    return {"scene_mean_nll_difference_a_minus_b": float(differences.mean()),
            "ci_95": np.quantile(sampled, [.025, .975]).tolist(), "common_scenes": len(common),
            "bootstrap_p_two_sided_centered": float(p), "bootstrap_repetitions": repetitions,
            "estimand": "equally weighted common scene mean NLL difference"}


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.zeros(len(order))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted.tolist()
