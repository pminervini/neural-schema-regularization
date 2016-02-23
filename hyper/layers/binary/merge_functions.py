# -*- coding: utf-8 -*-

from keras import backend as K
import sys


def translating_merge_function(args, similarity):
    """
    Keras Merge function for the Translating Embeddings model described in:
        A Bordes et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 201
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    relation_embedding, entity_embeddings = args[0], args[1]

    pred = relation_embedding[:, 0, :]
    subj, obj = entity_embeddings[:, 0, :], entity_embeddings[:, 1, :]

    translation = subj + pred
    nb_samples = K.shape(translation)[0]

    sim = K.reshape(similarity(translation, obj), (nb_samples, 1))

    return sim


def scaling_merge_function(args, similarity):
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
    subj, obj = entity_embeddings[:, 0, :], entity_embeddings[:, 1, :]

    scaling = subj * pred
    nb_samples = K.shape(scaling)[0]

    sim = K.reshape(similarity(scaling, obj), (nb_samples, 1))

    return sim

from hyper.layers import operations


def holographic_merge_function(args, similarity):
    relation_embedding, entity_embeddings = args[0], args[1]

    pred = relation_embedding[:, 0, :]
    subj, obj = entity_embeddings[:, 0, :], entity_embeddings[:, 1, :]

    scaling = subj * pred
    nb_samples = K.shape(scaling)[0]

    sim = K.reshape(similarity(scaling, obj), (nb_samples, 1))

    return sim


# aliases
TransE = TranslatingEmbeddings = translating_merge_function
ScalE = ScalingEmbeddings = scaling_merge_function


def get_function(function_name):
    this_module = sys.modules[__name__]
    if hasattr(this_module, function_name):
        function = getattr(this_module, function_name)
    else:
        raise ValueError("Unknown merge function: %s" % function_name)
    return function
