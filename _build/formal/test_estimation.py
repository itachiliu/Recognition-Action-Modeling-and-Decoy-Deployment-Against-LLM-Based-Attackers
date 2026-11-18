"""Independent numerical, leakage and clustering checks; synthetic data only."""
import itertools
import unittest

import numpy as np

import estimation as est
import design
from simulation import synthetic


class NumericalChecks(unittest.TestCase):
    def test_variable_choice_gradient_and_hessian(self):
        rng = np.random.default_rng(52)
        f = rng.normal(size=(12, 7, 4))
        mask = np.ones((12, 7), dtype=bool)
        mask[::2, -2:] = False
        f[~mask] = 0
        y = np.array([rng.integers(m.sum()) for m in mask])
        w = rng.integers(0, 3, size=12).astype(float)
        t = rng.normal(size=4)
        objective = lambda value: est.choice_objective(value, f, y, mask, w, .013, True)
        loss, gradient, hessian = objective(t)
        numeric_gradient, numeric_hessian = [], []
        for i in range(4):
            delta = np.eye(4)[i] * 1e-5
            plus, minus = objective(t + delta), objective(t - delta)
            numeric_gradient.append((plus[0] - minus[0]) / 2e-5)
            numeric_hessian.append((plus[1] - minus[1]) / 2e-5)
        np.testing.assert_allclose(gradient, numeric_gradient, atol=1e-8)
        np.testing.assert_allclose(hessian, np.array(numeric_hessian).T, atol=1e-8)

    def test_recognition_gradient_and_hessian(self):
        rng = np.random.default_rng(51)
        z = np.c_[np.ones(20), rng.normal(size=(20, 4))]
        y = rng.integers(0, 2, 20)
        w = rng.integers(0, 3, 20).astype(float)
        theta = rng.normal(size=5)
        objective = lambda t: est.logistic_objective(t, z, y, w, .02)
        _, gradient, hessian = objective(theta)
        for j in range(5):
            d = np.eye(5)[j] * 1e-5
            p, m = objective(theta + d), objective(theta - d)
            self.assertAlmostEqual(gradient[j], (p[0] - m[0]) / 2e-5, places=8)
            np.testing.assert_allclose(hessian[:, j], (p[1] - m[1]) / 2e-5, atol=1e-8)

    def test_marginal_independent_enumeration_and_nesting(self):
        rng = np.random.default_rng(53)
        f = rng.normal(size=(3, 10))
        alpha = rng.normal(size=5)
        gamma = rng.normal(size=10)
        beta = -.8
        p = est.sigmoid(np.c_[np.ones(3), f[:, 6:]] @ alpha)
        expected = np.zeros(3)
        for bits in itertools.product([0, 1], repeat=3):
            bits = np.asarray(bits)
            s = np.exp(f @ gamma + beta * bits)
            expected += np.prod(p ** bits * (1 - p) ** (1 - bits)) * s / s.sum()
        np.testing.assert_allclose(est.exact_marginal(f, alpha, gamma, beta), expected, atol=1e-12)
        np.testing.assert_allclose(est.exact_marginal(f, alpha, gamma, 0),
                                   est.probabilities(f @ gamma, np.ones(3, dtype=bool)), atol=1e-12)

    def test_bootstrap_keeps_whole_scene_and_sample_count(self):
        data = synthetic()
        weights = est.bootstrap_weights(data, np.random.default_rng(54))
        self.assertEqual(weights.sum(), len(weights))
        for scene in np.unique(data["scene"]):
            self.assertEqual(len(set(weights[data["scene"] == scene])), 1)

    def test_prediction_cannot_use_test_self_report(self):
        data = synthetic()
        test = est.subset(data, data["split"] == "test")
        fitted = {"model": "RAG_QR", "alpha": [-2, .9, 1.3, 1., 1.1],
                  "gamma": [1.2, .6, .4, .2, .35, .25, .25, -.1, .15, -.2], "beta": -.7}
        before = est.predict(test, fitted)
        test["r"] = 1 - test["r"]
        test["choice"][:] = 0
        np.testing.assert_array_equal(before, est.predict(test, fitted))
        self.assertTrue(np.allclose(before.sum(axis=1), 1))

    def test_scene_design_isolated_and_balanced(self):
        scenes = design.build_scenes()
        result = design.design_audit(scenes, design.build_presentations(scenes))
        self.assertEqual(result["decoy_instances"], 240)
        self.assertGreaterEqual(result["decoy_cell_min"], 3)
        self.assertFalse(result["formal_ready"])
        self.assertTrue(all(s["scene_id"].startswith("FMS") for s in scenes))

    def test_unidentified_rho_is_not_filled_with_optimizer_initial_value(self):
        data = synthetic()
        data["r"][:] = 0
        with self.assertRaisesRegex(ValueError, "not identified"):
            est.fit_choice(data, .001, np.ones(len(data["choice"])), "RAG_QR")
        with self.assertRaisesRegex(ValueError, "only one observed label class"):
            est.fit_recognition(data, .001, np.ones(len(data["choice"])))

    def test_response_mapping_survives_both_id_and_position_permutations(self):
        scenes = design.build_scenes()[:1]
        items = design.build_presentations(scenes)
        records = []
        for item in items:
            records.append({
                "task_id": f"test|{item['scene_id']}|{item['layout']}|0",
                "scene_id": item["scene_id"], "layout": item["layout"], "ok": True,
                "prompt_sha256": item["prompt_sha256"],
                "canonical_choice": "C02", "canonical_suspicious": ["C01", "C03"],
            })
        data = est.dataset_from_records(items, records, {scenes[0]["scene_id"]: "train"})
        np.testing.assert_array_equal(data["F"][0], data["F"][1])
        np.testing.assert_array_equal(data["r"][0], data["r"][1])
        np.testing.assert_array_equal(data["choice"], [1, 1])
        records.append(dict(records[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            est.dataset_from_records(items, records, {scenes[0]["scene_id"]: "train"})


if __name__ == "__main__":
    unittest.main()
