#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, TimeDistributedMerge
from keras.models import Graph

from hyper.preprocessing import kb

import sys
import logging
import argparse

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2015'


def experiment(train_sequences, nb_entities, nb_predicates,
               entity_embedding_size=100, predicate_embedding_size=100):

    model = Graph()

    model.add_input(name='entity_idx', input_shape=(None,), dtype='int32')
    model.add_input(name='predicate_idx', input_shape=(1,), dtype='int32')

    entity_embedding_layer = Embedding(input_dim=nb_entities, output_dim=entity_embedding_size)
    predicate_embedding_layer = Embedding(input_dim=nb_predicates, output_dim=predicate_embedding_size)

    model.add_node(entity_embedding_layer, name='entity_embedding', input='entity_idx')
    model.add_node(predicate_embedding_layer, name='predicate_embedding', input='predicate_idx')

    model.add_node(layer=Activation('linear'), name='concat',
                   inputs=['predicate_embedding', 'entity_embedding'],
                   merge_mode='concat', concat_axis=1)

    model.add_output(name='output', input='concat')

    return


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Latent Factor Models for Knowledge Hypergraphs', formatter_class=formatter)
    argparser.add_argument('--train', required=True, type=argparse.FileType('r'))

    argparser.add_argument('--seed', action='store', type=int, default=1, help='Seed for the PRNG')

    argparser.add_argument('--entity-embedding-size', action='store', type=int, default=100,
                           help='Size of entity embeddings')
    argparser.add_argument('--predicate-embedding-size', action='store', type=int, default=None,
                           help='Size of predicate embeddings')

    args = argparser.parse_args(argv)

    np.random.seed(args.seed)

    train_facts = list()
    for line in args.train:
        subject, predicate, object = line.split()
        train_facts.append(kb.Fact(predicate_name=predicate, argument_names=[subject, object]))

    parser = kb.KnowledgeBaseParser(train_facts)

    nb_entities, nb_predicates = len(parser.entity_vocabulary), len(parser.predicate_vocabulary)

    train_sequences = parser.facts_to_sequences(train_facts)

    experiment(train_sequences, nb_entities, nb_predicates,
               entity_embedding_size=args.entity_embedding_size,
               predicate_embedding_size=args.predicate_embedding_size)
    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
