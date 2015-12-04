# -*- coding: utf-8 -*-

import numpy as np


def pad_sequences(sequences, max_len=None, padding_pre=True, truncating_pre=True, value=0, dtype='int32'):
    """
    Pad each sequence to the same length (the length of the longest sequence, unless specified otherwise)
    If max_len is provided, any sequence longer than max_len is truncated to max_len.
    Truncation happens off either the beginning (default) or the end of the sequence.

    :param sequences: Input sequences.
    :param max_len: Maximum length of the output sequences (default: None).
    :param padding_pre:
    :param truncating_pre:
    :param value: Value to add to sequences for padding them to the same length.
    :param dtype:
    :return:
    """

    sequence_lengths = [len(seq) for seq in sequences]

    nb_samples = len(sequences)
    max_len = np.max(sequence_lengths) if max_len is None else max_len

    x = np.full((nb_samples, max_len), value).astype(dtype)

    for idx, seq in enumerate(sequences):
        if len(seq) > 0:
            trunc = seq[- max_len:] if truncating_pre is True else seq[:max_len]
            if padding_pre is True:
                x[idx, -len(trunc):] = trunc
            else:
                x[idx, :len(trunc)] = trunc

    return x
