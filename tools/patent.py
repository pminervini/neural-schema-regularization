#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import h5py
import pickle

import sys
import logging


def read_triples(path):
    with open(path, 'rt') as f:
        lines = f.readlines()
    triples = [(s.strip(), p.strip(), o.strip()) for [s, p, o] in [l.split('\t') for l in lines]]
    return triples


def main(argv):
    test_triples = read_triples('data/wn18/wordnet-mlj12-test.txt')

    definition_triples = read_triples('data/wn18/wordnet-mlj12-definitions.txt')
    entity_to_name = {entity: name for (entity, name, definition) in definition_triples}

    weights_path = 'model_farm/wn18_transe_200_l1/wn18_weights.h5'
    parser_path = 'model_farm/wn18_transe_200_l1/wn18_parser.p'

    weights_s_path = 'model_farm/wn18_transe_200_l1_100000/wn18_weights.h5'
    parser_s_path = 'model_farm/wn18_transe_200_l1_100000/wn18_parser.p'

    predicates = ['_part_of', '_has_part']

    with h5py.File(weights_path) as f:
        E = f['/embedding_2/embedding_2_W'][()]
        W = f['/embedding_1/embedding_1_W'][()]

    with h5py.File(weights_s_path) as f:
        E_s = f['/embedding_2/embedding_2_W'][()]
        W_s = f['/embedding_1/embedding_1_W'][()]

    with open(parser_path, 'rb') as f:
        parser = pickle.load(f)

    with open(parser_s_path, 'rb') as f:
        parser_s = pickle.load(f)

    entity_index = parser.entity_index
    predicate_index = parser.predicate_index

    entity_index_s = parser_s.entity_index
    predicate_index_s = parser_s.predicate_index

    for s, p, o in [(s, p, o) for (s, p, o) in test_triples if p in predicates]:
        # q is the inverse of p (e.g. if p is has_part, q is part_of)
        q = predicates[1] if p == predicates[0] else predicates[0]

        s_idx, o_idx = entity_index[s], entity_index[o]
        p_idx, q_idx = predicate_index[p], predicate_index[q]

        s_idx_s, o_idx_s = entity_index_s[s], entity_index_s[o]
        p_idx_s, q_idx_s = predicate_index_s[p], predicate_index_s[q]

        # Score for s, p, o according to the schema-less model
        score = - np.sum(np.abs(E[s_idx, :] + W[p_idx, :] - E[o_idx, :]))
        # Score for o, q, s according to the schema-less model
        i_score = - np.sum(np.abs(E[o_idx, :] + W[q_idx, :] - E[s_idx, :]))

        # Score for s, p, o according to the schema-aware model
        score_s = - np.sum(np.abs(E_s[s_idx_s, :] + W_s[p_idx_s, :] - E_s[o_idx_s, :]))

        # Score for o, q, s according to the schema-aware model
        i_score_s = - np.sum(np.abs(E_s[o_idx_s, :] + W_s[q_idx_s, :] - E_s[s_idx_s, :]))

        if abs(score - i_score) - abs(score_s - i_score_s) > 1.8:
            print()
            print('[No Schema] score(', entity_to_name[s], p, entity_to_name[o], ') = ', score)
            print('[No Schema] score(', entity_to_name[o], q, entity_to_name[s], ') = ', i_score)
            print('[Schema] score(', entity_to_name[s], p, entity_to_name[o], ') = ', score_s)
            print('[Schema] score(', entity_to_name[o], q, entity_to_name[s], ') = ', i_score_s)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
