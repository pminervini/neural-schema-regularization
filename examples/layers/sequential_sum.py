# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda

import logging
import sys


def func(X):
    return K.sum(X, axis=1)


def main(argv):
    model = Sequential()

    W = np.ones((3, 2))

    model.add(Embedding(input_dim=3, output_dim=2, weights=[W]))
    model.add(Lambda(func))

    model.compile(loss='binary_crossentropy', optimizer='sgd')

    X = np.asarray([[0, 1, 0, 2]])
    Y = model.predict(X, batch_size=1)[0]

    print(Y)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
