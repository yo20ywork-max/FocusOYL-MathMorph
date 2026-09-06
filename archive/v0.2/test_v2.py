import unittest
import numpy as np
from spectral import transform,components,transform_from_components
from bench_v2 import parse_answer,paired

class MathematicalTests(unittest.TestCase):
    def test_constraints(self):
        for seed in range(30):
            w=np.random.default_rng(seed).normal(size=(17,32)).astype(np.float32)
            for direction in [-1,1]:
                z,s=transform(w,rank=4,seed=seed,direction=direction)
                self.assertAlmostEqual(float(np.linalg.norm(z)/np.linalg.norm(w)),1.,places=5)
                self.assertLessEqual(s['relative_change'],.050002)
                self.assertEqual(z.shape,w.shape)
                self.assertEqual(np.linalg.matrix_rank(z),np.linalg.matrix_rank(w))
                self.assertLess(abs(s['relative_change']-s['theoretical_relative_change']),1e-5)
    def test_zero(self):
        z,s=transform(np.zeros((4,8)));self.assertFalse(z.any());self.assertTrue(s['no_effect'])
    def test_rank_full(self):
        w=np.eye(4,dtype=np.float32);z,s=transform(w,rank=4);np.testing.assert_array_equal(z,w)
    def test_tensor_shape(self):
        w=np.random.default_rng(2).normal(size=(3,4,8));z,s=transform(w,rank=2);self.assertEqual(w.shape,z.shape)
    def test_no_uniform_improvement(self):
        w=np.random.default_rng(4).normal(size=(8,16));z,s=transform(w,rank=2)
        x=np.ones(16);truth=w@x
        self.assertEqual(np.linalg.norm(w@x-truth),0)
        self.assertGreater(np.linalg.norm(z@x-truth),0)
    def test_scale(self):
        w=np.random.default_rng(2).normal(size=(8,16));z,_=transform(w,rank=2)
        zz,_=transform(2*w,rank=2);np.testing.assert_allclose(zz,2*z,rtol=2e-5,atol=1e-5)
    def test_invalid(self):
        for w in [np.ones(4),np.full((4,8),np.nan)]:
            with self.assertRaises(ValueError):transform(w)
    def test_grader_no_expected_leak(self):
        c={'check':'mcq','expected':'C'}
        self.assertEqual(parse_answer('FINAL: B',c),'B')
        self.assertIsNone(parse_answer('A and C are possible',c))
        self.assertEqual(parse_answer('The answer is C.',c),'C')
    def test_integer(self):
        c={'check':'integer','expected':'13'}
        self.assertEqual(parse_answer('FINAL: -4',c),'-4')
        self.assertIsNone(parse_answer('13 or 14',c))
    def test_stats(self):
        x=paired([False]*8,[True]*8);self.assertEqual(x['one_sided_exact_p'],1/256)
        self.assertEqual(paired([True]*5,[True]*5)['one_sided_exact_p'],1)
if __name__=='__main__':unittest.main(verbosity=2)
