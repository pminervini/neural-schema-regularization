# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter

import logging


def create_mask(nb_items, embedding_size, embedding_lengths):
    assert nb_items == len(embedding_lengths)

    mask = np.zeros((embedding_size, nb_items))

    for item_idx, embedding_length in enumerate(embedding_lengths):
        mask[:embedding_length, item_idx] = 1

    return mask


def get_embedding_lengths(triples, cutpoints, embedding_lengths):
    entity_seq = [s for (s, _, _) in triples] + [o for (_, _, o) in triples]

    cutpoints = sorted(cutpoints)

    entity_lengths = {}
    entity_counts = sorted(Counter(entity_seq).items(), key=lambda entry: entry[1])

    for (entity, count) in entity_counts:
        idx = next((cutpoints.index(c) for c in cutpoints if c > count), len(cutpoints))
        entity_lengths[entity] = embedding_lengths[idx - 1]

    return entity_lengths
