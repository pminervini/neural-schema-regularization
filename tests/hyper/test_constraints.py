# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K
import hyper.layers.core
from hyper.constraints import norm, mask
import unittest


class TestConstraints(unittest.TestCase):

    def setUp(self):
        self.rs = np.random.RandomState(1)

    def test_unitnorm(self):
        for _ in range(64):
            M, N = 32, 32

            x = self.rs.rand()

            example_array = self.rs.rand(M, N) * 100. - 50.
            example_array[0, 0] = 0.

            unitnorm_instance = norm(m=x, axis=0)
            normalized = unitnorm_instance(K.variable(example_array))
            normalized_value = K.eval(normalized)

            norm_of_normalized = np.sqrt(np.sum(normalized_value ** 2, axis=0))

            # in the unit norm constraint, it should be equal to x
            difference = norm_of_normalized - x

            largest_difference = np.max(np.abs(difference))
            self.assertTrue(np.abs(largest_difference) < 1e-5)

    def test_mask(self):
        for _ in range(64):
            M, N, S, R = 32, 32, 16, 4

            example_array = self.rs.rand(M, N) * 100. - 50.
            example_array[0, 0] = 0.

            mask_value = np.zeros((M, N))
            mask_value[R, :S] = 1

            mask_instance = mask(mask=mask_value)
            masked = mask_instance(K.variable(example_array))

            masked_value = K.eval(masked)

            for i in range(S):
                self.assertTrue(abs(masked_value[R, i]) > masked_value[R, S + i])

if __name__ == '__main__':
    unittest.main()
