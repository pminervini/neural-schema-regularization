# -*- coding: utf-8 -*-

import unittest
import numpy as np

from hyper.layers import recurrent
from keras import backend as K

nb_samples, timesteps, input_dim, output_dim = 3, 3, 10, 10


def _runner(layer_class):
    """
    All the recurrent layers share the same interface, so we can run through them with a single function.
    """
    for ret_seq in [True, False]:
        layer = layer_class(return_sequences=ret_seq, input_shape=(timesteps, input_dim))
        layer.input = K.variable(np.ones((nb_samples, timesteps, input_dim)))
        layer.get_config()

        for train in [True, False]:
            out = K.eval(layer.get_output(train))
            if ret_seq:
                assert(out.shape == (nb_samples, timesteps, output_dim))
            else:
                assert(out.shape == (nb_samples, output_dim))
            mask = layer.get_output_mask(train)


class TestRNNS(unittest.TestCase):
    """
    Test all the Recurrent Neural Network models using a generic test runner function defined above.
    """
    def test_simple(self):
        _runner(recurrent.RecurrentTransE)

if __name__ == '__main__':
    unittest.main()
