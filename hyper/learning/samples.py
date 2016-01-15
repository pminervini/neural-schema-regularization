# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from keras.preprocessing import sequence
from keras.datasets import imdb

import logging
import sys


class IndexGenerator(metaclass=ABCMeta):
    @abstractmethod
    def generate(self, n_samples, indices):
        pass


class UniformRandomIndexGenerator(IndexGenerator):
    def __init__(self, random_state):
        """
        Initializes the object.
        :param random_state: numpy.random.RandomState instance.
        """
        self.random_state = random_state

    def generate(self, n_samples, indices):
        """
        Creates a NumPy vector of 'n_samples', randomly selected by 'indices'.
        :param n_samples: Number of samples to generate.
        :param indices: List or NumPy vector containing the candidate indices.
        :return:
        """
        if isinstance(indices, list):
            indices = np.array(indices)

        rand_ints = self.random_state.random_integers(0, indices.size - 1, n_samples)
        return indices[rand_ints]



def main(argv):
    max_features = 5000
    maxlen = 200
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    print(X_train[0, :])

    rs = np.random.RandomState(seed=0)
    ig = UniformRandomIndexGenerator(rs)

    print(ig.generate(10, [1, 3, 5]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
