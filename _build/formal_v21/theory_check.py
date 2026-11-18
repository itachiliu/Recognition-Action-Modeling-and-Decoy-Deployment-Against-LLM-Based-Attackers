"""Numerical verification of the paper's recognition-channel proposition."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "theory_check.json"


def q_focus(rho, p_focus, utilities, recognition_probabilities):
    """Exact focus probability when only p_focus is varied."""
    utilities = np.asarray(utilities, dtype=float)
    probabilities = np.asarray(recognition_probabilities, dtype=float)
    total = 0.0
    for bits in itertools.product((0, 1), repeat=len(utilities) - 1):
        weight = 1.0
        exponents = []
        for bit, probability in zip(bits, probabilities):
            weight *= probability if bit else 1 - probability
            exponents.append(rho if bit else 1.0)
        base = sum(multiplier * math.exp(utility) for multiplier, utility in zip(exponents, utilities[1:]))
        focus_unrecognized = 1.0 / (1.0 + base)
        focus_recognized = rho / (rho + base)
        total += weight * ((1 - p_focus) * focus_unrecognized + p_focus * focus_recognized)
    return total


def main():
    utilities = [0.0, 1.0, 0.3]
    probabilities = [0.3, 0.7]
    rows = []
    for rho in (0.01, 0.1, 0.25, 0.5, 0.75, 1.0):
        p = 0.5
        eps = 1e-6
        finite_difference = (
            q_focus(rho, p + eps, utilities, probabilities)
            - q_focus(rho, p - eps, utilities, probabilities)
        ) / (2 * eps)
        expected_derivative = -(
            q_focus(rho, p - 1e-8, utilities, probabilities)
            - q_focus(rho, p, utilities, probabilities)
        ) / 1e-8
        rows.append(
            {
                "rho": rho,
                "finite_difference_dq_dp": finite_difference,
                "numerical_G": -finite_difference,
                "rho_one_q": q_focus(1.0, p, utilities, probabilities),
            }
        )
    valid = all(row["finite_difference_dq_dp"] <= 1e-12 for row in rows)
    monotone = all(
        rows[i]["numerical_G"] >= rows[i + 1]["numerical_G"] - 1e-12
        for i in range(len(rows) - 1)
    )
    result = {
        "description": "Fixed utilities and other-target recognition probabilities; focus recognition probability varied by finite difference.",
        "rows": rows,
        "dq_dp_nonpositive": valid,
        "G_nonincreasing_in_rho": monotone,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
