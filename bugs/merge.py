#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Merge

import sys


flag = sys.argv[1] if len(sys.argv) > 1 else 'c'


def loss(y_true, y_pred):
    loss_value = K.sum(y_true) + K.sum(y_pred)
    return loss_value

a_encoder = Sequential()
a_embedding_layer = Embedding(input_dim=1, output_dim=1, init='zero', input_length=1)
a_encoder.add(a_embedding_layer)
a_encoder.add(Flatten())

if flag == 'a':
    a_encoder.compile(loss=loss, optimizer='adagrad')

    x = [np.array([[0]])]
    y = np.array([[0]])

    print(a_encoder.get_weights())
    a_encoder.fit(x=x, y=y, nb_epoch=1, batch_size=1, shuffle=False, verbose=0)
    print(a_encoder.get_weights())
elif flag == 'b':
    b_encoder = Sequential()
    b_embedding_layer = Embedding(input_dim=1, output_dim=1, init='zero', input_length=1)
    b_encoder.add(b_embedding_layer)
    b_encoder.add(Flatten())

    model_one = Sequential()
    model_one.add(Merge([a_encoder, b_encoder]))
    model_one.compile(loss=loss, optimizer='adagrad')

    x = [np.array([[0]]), np.array([[0]])]
    y = np.array([[0]])

    print(model_one.get_weights())
    model_one.fit(x=x, y=y, nb_epoch=1, batch_size=1, shuffle=False, verbose=0)
    print(model_one.get_weights())
else:
    b_encoder = Sequential()
    b_embedding_layer = Embedding(input_dim=1, output_dim=1, init='zero', input_length=1)
    b_encoder.add(b_embedding_layer)
    b_encoder.add(Flatten())

    model_one = Sequential()
    model_one.add(Merge([a_encoder, b_encoder]))

    c_encoder = Sequential()
    c_embedding_layer = Embedding(input_dim=1, output_dim=1, init='zero', input_length=1)
    c_encoder.add(c_embedding_layer)
    c_encoder.add(Flatten())

    model_two = Sequential()
    model_two.add(Merge([model_one, c_encoder]))
    model_two.compile(loss=loss, optimizer='adagrad')

    x = [np.array([[0]]), np.array([[0]]), np.array([[0]])]
    y = np.array([[0]])

    print(model_two.get_weights())
    model_two.fit(x=x, y=y, nb_epoch=1, batch_size=1, shuffle=False, verbose=0)
    print(model_two.get_weights())
