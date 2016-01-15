#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import LambdaMerge

from keras import backend as K

from hyper.preprocessing import knowledgebase

import sys
import logging
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2015'


def merge_function(args):
    relation_embedding = args[0]
    entity_embeddings = args[1]
    return relation_embedding[:, 0, :] * entity_embeddings[:, 0, :] * entity_embeddings[:, 1, :]


def experiment(train_sequences, nb_entities, nb_predicates,
               entity_embedding_size=100, predicate_embedding_size=100):

    predicate_encoder = Sequential()
    entity_encoder = Sequential()

    predicate_embedding_layer = Embedding(input_dim=nb_predicates, output_dim=predicate_embedding_size, input_length=1)
    predicate_encoder.add(predicate_embedding_layer)

    entity_embedding_layer = Embedding(input_dim=nb_entities, output_dim=entity_embedding_size, input_length=None)
    entity_encoder.add(entity_embedding_layer)

    model = Sequential()
    merge_layer = LambdaMerge([predicate_encoder, entity_encoder], function=merge_function)
    model.add(merge_layer)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print(train_sequences)
    Xr = np.array([[1]])
    Xe = np.array([[1, 2]])

    model.predict([Xr, Xe], batch_size=1)

    return


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Latent Factor Models for Knowledge Hypergraphs', formatter_class=formatter)
    argparser.add_argument('--train', required=True, type=argparse.FileType('r'))

    argparser.add_argument('--seed', action='store', type=int, default=1, help='Seed for the PRNG')

    argparser.add_argument('--entity-embedding-size', action='store', type=int, default=100,
                           help='Size of entity embeddings')
    argparser.add_argument('--predicate-embedding-size', action='store', type=int, default=100,
                           help='Size of predicate embeddings')

    args = argparser.parse_args(argv)

    np.random.seed(args.seed)

    train_facts = []
    for line in args.train:
        subj, pred, obj = line.split()
        train_facts += [knowledgebase.Fact(predicate_name=pred, argument_names=[subj, obj])]

    parser = knowledgebase.KnowledgeBaseParser(train_facts)

    nb_entities = len(parser.entity_vocabulary) + 1
    nb_predicates = len(parser.predicate_vocabulary) + 1

    entity_embedding_size = args.entity_embedding_size
    predicate_embedding_size = args.predicate_embedding_size

    train_sequences = parser.facts_to_sequences(train_facts)

    experiment(train_sequences, nb_entities, nb_predicates,
               entity_embedding_size=entity_embedding_size,
               predicate_embedding_size=predicate_embedding_size)
    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
