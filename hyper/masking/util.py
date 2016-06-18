# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter

import logging


def create_mask(nb_items, embedding_size, embedding_lengths):
    assert nb_items == len(embedding_lengths)

    mask = np.zeros((nb_items, embedding_size))

    for item_idx, embedding_length in enumerate(embedding_lengths):
        mask[item_idx, :embedding_length] = 1

    return mask


def get_embedding_lengths(triples, cut_points, embedding_lengths):
    entity_seq = [s for (s, _, _) in triples] + [o for (_, _, o) in triples]

    entity_lengths = {}
    entity_counts = sorted(Counter(entity_seq).items(), key=lambda entry: entry[1])

    for (entity, count) in entity_counts:
        idx = next((cut_points.index(c) for c in cut_points if c >= count), len(cut_points))
        entity_lengths[entity] = embedding_lengths[idx]

    return entity_lengths
