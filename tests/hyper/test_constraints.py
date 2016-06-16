# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K
import hyper.layers.core
from hyper.constraints import norm
import unittest


class TestConstraints(unittest.TestCase):

    def setUp(self):
        pass

    def test_unitnorm(self):
        rs = np.random.RandomState(1)

        for _ in range(2):
            x = rs.rand()

            example_array = np.random.random((100, 100)) * 100. - 50.
            example_array[0, 0] = 0.

            unitnorm_instance = norm(m=x, axis=0)
            normalized = unitnorm_instance(K.variable(example_array))
            norm_of_normalized = np.sqrt(np.sum(K.eval(normalized) ** 2, axis=0))

            # in the unit norm constraint, it should be equal to x
            difference = norm_of_normalized - x

            largest_difference = np.max(np.abs(difference))
            self.assertTrue(np.abs(largest_difference) < 10e-5)

if __name__ == '__main__':
    unittest.main()
