"""Numerical rejection boundary; no trained-model accuracy is tested here."""
import unittest
import numpy as np
from ilm.training_only.frozen_encoder_verifier import accepted_in_feature_ball


class FeatureBallTest(unittest.TestCase):
    def test_strict_boundary_and_collapsed_features(self):
        np.testing.assert_array_equal(accepted_in_feature_ball([0,.49,.5,.51,0],[1,1,1,1,0]),
                                      [True,True,False,False,False])

    def test_invalid_distance_rejected(self):
        for d,s in [([float('nan')],[1]),([-1],[1]),([0],[-1]),([0],[1,2])]:
            with self.assertRaises(ValueError):accepted_in_feature_ball(d,s)


if __name__=='__main__':unittest.main()
