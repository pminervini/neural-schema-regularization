#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.constraints import MaxNorm
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=500, test_split=0.05)
X_test = sequence.pad_sequences(X_test, maxlen=100)

threshold = 0.01
for norm_constraint in [None, MaxNorm(m=threshold, axis=1)]:
    np.random.seed(1337)

    model = Sequential()

    embedding_layer = Embedding(500, 10, input_length=100, W_constraint=norm_constraint)
    model.add(embedding_layer)
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_test, y_test, batch_size=32, nb_epoch=5, validation_data=(X_test, y_test), verbose=2)

    W = embedding_layer.trainable_weights[0].get_value()

    if norm_constraint is not None:
        for i in range(500):
            assert np.linalg.norm(W[i, :]) - threshold < 0
