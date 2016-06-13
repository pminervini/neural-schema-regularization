#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Merge


def loss(y_true, y_pred):
    loss_value = K.sum(y_true) + K.sum(y_pred)
    return loss_value


def create_encoder(identifier):
    encoder = Sequential()
    embedding_layer = Embedding(input_dim=1, output_dim=1, init='zero',
                                input_length=1, trainable=(identifier != 1))
    encoder.add(embedding_layer)
    encoder.add(Flatten())
    return encoder

a_encoder = create_encoder(1)
b_encoder = create_encoder(2)
c_encoder = create_encoder(3)

model_one = Sequential()
model_one.add(Merge([a_encoder, b_encoder]))

model_two = Sequential()
model_two.add(Merge([model_one, c_encoder]))
model_two.compile(loss=loss, optimizer='adagrad')

x = [np.array([[0]]), np.array([[0]]), np.array([[0]])]
y = np.array([[0]])

print('Weights before update: %s' % model_two.get_weights())
model_two.fit(x=x, y=y, nb_epoch=1, batch_size=1, shuffle=False, verbose=0)
print('Weights after update:  %s' % model_two.get_weights())

for layer in model_two.trainable_weights:
    print(layer)
