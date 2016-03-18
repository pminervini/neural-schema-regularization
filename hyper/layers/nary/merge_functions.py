# -*- coding: utf-8 -*-

import theano.tensor as T
from keras import backend as K

import sys


def recurrent_translating_merge_function(args, similarity):
    """
    Keras Merge function for the Translating Embeddings model described in:
        A Bordes et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 2013
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    relation_embedding, entity_embeddings = args[0], args[1]
    pred = relation_embedding[:, 0, :]
    translations = T.extra_ops.cumsum(entity_embeddings, axis=1)[:, 0, :]
    sim = K.reshape(similarity(translations, pred), (-1, 1))
    return sim


def recurrent_scaling_merge_function(args, similarity):
    """
    Keras Merge function for the multiplicative interactions model described in:
        B Yang et al. - Embedding Entities and Relations for Learning and Inference in Knowledge Bases - ICLR 2015
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the scaled subject embedding, and the object embedding.
    """
    relation_embedding, entity_embeddings = args[0], args[1]
    pred = relation_embedding[:, 0, :]
    scalings = T.extra_ops.cumprod(entity_embeddings, axis=1)[:, 0, :]
    sim = K.reshape(similarity(scalings, pred), (-1, 1))
    return sim


# aliases
rTransE = RecurrentTranslatingEmbeddings = recurrent_translating_merge_function
rScalE = RecurrentScalingEmbeddings = recurrent_scaling_merge_function


def get_function(function_name):
    this_module = sys.modules[__name__]
    if hasattr(this_module, function_name):
        function = getattr(this_module, function_name)
    else:
        raise ValueError("Unknown merge function: %s" % function_name)
    return function
