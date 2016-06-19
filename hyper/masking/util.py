# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter

from keras import backend as K

import logging


def create_mask(nb_items, embedding_size, mask_ranges):
    assert nb_items == mask_ranges.shape[0]
    mask = np.zeros((nb_items, embedding_size), dtype=K.floatx())
    for i in range(mask_ranges.shape[0]):
        mask[i, mask_ranges[i, 0]:mask_ranges[i, 1]] = 1.
    return mask


def get_entity_bins(triples, cut_points):
    entity_seq = [s for (s, _, _) in triples] + [o for (_, _, o) in triples]
    entity_counts = sorted(Counter(entity_seq).items(), key=lambda entry: entry[1])

    entity_bins = {}
    for (entity, count) in entity_counts:
        bin_idx = next((cut_points.index(c) for c in cut_points if c >= count), len(cut_points))
        entity_bins[entity] = bin_idx

    return entity_bins
