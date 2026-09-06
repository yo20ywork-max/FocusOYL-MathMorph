"""Synthetic algebra and archive-contract checks; no model inference."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive" / "v0.4"))
from convert_prism import BASELINE_SHA256, RECIPE
from geometry import bounded_shrink, metric_tail_direction
from anchor_protection import protect_delta

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(917)
        self.w = self.rng.normal(size=(9, 13)).astype(np.float32)
        u, _, _ = np.linalg.svd(self.w, full_matrices=False)
        self.p = u[:, :4] @ u[:, :4].T
        self.tail = self.w - self.p @ self.w

    def test_projector(self):
        np.testing.assert_allclose(self.p @ self.p, self.p, atol=2e-6)
        np.testing.assert_allclose(self.p.T, self.p, atol=2e-6)

    def test_orthogonal_split(self):
        self.assertAlmostEqual(float(np.sum((self.p @ self.w) * self.tail)), 0, places=4)

    def test_weight_budget(self):
        z, stats = bounded_shrink(self.w, self.tail, 0.1176)
        self.assertLessEqual(stats["relative_change"], 0.11761)
        np.testing.assert_allclose(z, self.w - stats["alpha"] * self.tail, atol=1e-6)

    def test_zero_residual(self):
        z, stats = bounded_shrink(self.w, np.zeros_like(self.w), 0.12)
        np.testing.assert_array_equal(z, self.w)
        self.assertEqual(stats["alpha"], 0)

    def test_energy_relation(self):
        alpha = 0.2
        z = self.w - alpha * self.tail
        expected = np.linalg.norm(self.p @ self.w)**2 + (1-alpha)**2 * np.linalg.norm(self.tail)**2
        self.assertAlmostEqual(float(np.linalg.norm(z)**2), float(expected), places=4)

    def test_euclidean_direction(self):
        identity = np.eye(self.w.shape[0], dtype=np.float32)
        tail, _ = metric_tail_direction(self.w, None, identity, identity, 4)
        np.testing.assert_allclose(tail, self.tail, atol=1e-5)

    def test_contrast_protection(self):
        q, _ = np.linalg.qr(self.rng.normal(size=(9, 3)))
        protected = protect_delta(self.tail, q.astype(np.float32))
        np.testing.assert_allclose(q.T @ protected, 0, atol=2e-6)

    def test_recipe_identity(self):
        self.assertEqual(RECIPE, {"label":"euclid_control","layers":[23],"rank":512,"budget":0.12,"geometry":False})
        self.assertEqual(len(BASELINE_SHA256), 64)
        self.assertTrue((ROOT / "archive/v0.4/convert_geometry.py").is_file())

    def test_evidence_identity(self):
        manifest = json.loads((ROOT / "provenance/ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertGreater(len(manifest["files"]), 20)
        for record in manifest["files"]:
            self.assertTrue((ROOT / record["published_path"]).is_file())

if __name__ == "__main__":
    unittest.main()
