# -*- coding: utf-8 -*-

import numpy as np

from keras.preprocessing import sequence
from keras.datasets import imdb

import hyper.learning.samples as samples

import unittest


class TestSamples(unittest.TestCase):

    def setUp(self):
        pass

    def test_samples(self):

        for _ in range(1024):
            seed = np.random.random_integers(0, 8192)

            rs_a = np.random.RandomState(seed=seed)
            ig_a = samples.UniformRandomIndexGenerator(rs_a)

            rs_b = np.random.RandomState(seed=seed)
            ig_b = samples.UniformRandomIndexGenerator(rs_b)

            vs = np.random.random_integers(0, 8192, 1024)

            eq = ig_a.generate(1024, vs) == ig_b.generate(1024, vs)
            self.assertTrue(False not in eq)

    def test_imdb(self):
        max_features, max_length = 5000, 200

        (X_train, y_train), (X_valid, y_valid) = imdb.load_data(nb_words=max_features, test_split=0.2)

        X_train = sequence.pad_sequences(X_train, maxlen=max_length)
        X_valid = sequence.pad_sequences(X_valid, maxlen=max_length)

        self.assertTrue(X_train.shape == (20000, 200))
        self.assertTrue(X_valid.shape == (5000, 200))

if __name__ == '__main__':
    unittest.main()
