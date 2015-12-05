# -*- coding: utf-8 -*-

import numpy as np

from hyper.preprocessing import kb
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.constraints import unitnorm

from hyper.layers.recurrent import RecurrentTransE


def read_lines(fname):
    with open(fname) as f:
        content = f.readlines()
    return content

print("Loading data...")

training_lines = read_lines('data/fb15k/freebase_mtr100_mte100-train.txt')
training_facts = [line.split() for line in training_lines]

parser = kb.KnowledgeBaseParser(training_facts)

training_sequences = parser.facts_to_sequences(training_facts)
max_len = max([len(seq) for seq in training_sequences])

X = np.asmatrix(training_sequences, dtype='int32')
print(X.shape)

entities = list(set([seq[0] for seq in training_sequences] + [seq[2] for seq in training_sequences]))
predicates = list(set([seq[1] for seq in training_sequences]))

symbols = entities + predicates
nb_symbols = len(symbols)

model = Sequential()

emb_init = np.ones((nb_symbols, 100))
embedding_layer = Embedding(nb_symbols, 100, input_length=max_len, weights=[emb_init], W_constraint=unitnorm())
model.add(embedding_layer)


recurrent_layer = RecurrentTransE()
model.add(recurrent_layer)

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')

#model.fit([X[0]], [1], batch_size=1, nb_epoch=1)

print(model.states)

print(model.predict(X[0], batch_size=1))
