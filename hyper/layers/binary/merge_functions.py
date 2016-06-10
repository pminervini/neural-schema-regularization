# -*- coding: utf-8 -*-

import theano
from keras import backend as K
from hyper.layers import operations

from hyper import norms

import sys


def to_triples(args):
    relation_embedding, entity_embeddings = args[0], args[1]
    subj, pred, obj = entity_embeddings[:, 0, :], relation_embedding[:, 0, :], entity_embeddings[:, 1, :]
    return subj, pred, obj


def translating_merge_function(args, similarity):
    """
    Keras Merge function for the Translating Embeddings model described in:
        A Bordes et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 2013
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    translation = subj + pred
    sim = K.reshape(similarity(translation, obj), (-1, 1))
    return sim


def dual_translating_merge_function(args, similarity):
    """
    Keras Merge function for the Dual Translating Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    n = subj.shape[1]
    pred_subj = pred[:, :n]
    pred_obj = pred[:, n:]

    translation_subj = subj + pred_subj
    translation_obj = obj + pred_obj

    sim = K.reshape(similarity(translation_subj, translation_obj), (-1, 1))
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
    subj, pred, obj = to_triples(args)

    scaling = subj * pred
    sim = K.reshape(similarity(scaling, obj), (-1, 1))
    return sim


def dual_scaling_merge_function(args, similarity):
    """
    Keras Merge function for the Dual Scaling Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    n = subj.shape[1]
    pred_subj = pred[:, :n]
    pred_obj = pred[:, n:]

    scaling_subj = subj * pred_subj
    scaling_obj = obj * pred_obj

    sim = K.reshape(similarity(scaling_subj, scaling_obj), (-1, 1))
    return sim


def scaling_translating_merge_function(args, similarity):
    """
    Keras Merge function for the Dual Scaling Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    n = subj.shape[1]
    scaling = pred[:, :n]
    translation = pred[:, n:]

    transformation = (subj * scaling) + translation

    sim = K.reshape(similarity(transformation, obj), (-1, 1))
    return sim


def holographic_merge_function(args, similarity):
    """
    Keras Merge function for the multiplicative interactions model described in:
        M Nickel et al. - EHolographic Embeddings of Knowledge Graphs - AAAI 2016
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The dot product between between the predicate embedding, and the cross correlation
        of the subject and the object embeddings.
    """
    subj, pred, obj = to_triples(args)

    res, _ = theano.scan(lambda s, o: operations.circular_cross_correlation(s, o), sequences=[subj, obj])

    sim = K.reshape(similarity(pred, res), (-1, 1))
    return sim


def diagonal_affine_merge_function(args, similarity):
    """
    Keras Merge function for the Diagonal Affine Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    n = subj.shape[1]

    affine_transformation = (subj * pred[:, :n]) + pred[:, n:]
    sim = K.reshape(similarity(affine_transformation, obj), (-1, 1))
    return sim


def dual_diagonal_affine_merge_function(args, similarity):
    """
    Keras Merge function for the Dual Diagonal Affine Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    n = subj.shape[1]
    pred_subj = pred[:, (2 * n):]
    pred_obj = pred[:, :(2 * n)]

    affine_transformation_subj = (subj * pred_subj[:, :n]) + pred_subj[:, n:]
    affine_transformation_obj = (subj * pred_obj[:, :n]) + pred_obj[:, n:]

    sim = K.reshape(similarity(affine_transformation_subj, affine_transformation_obj), (-1, 1))
    return sim


def concatenate_merge_function(args, similarity):
    """
    Keras Merge function for the Concatenating Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    concatenation = K.concatenate([subj, obj], axis=1)
    sim = K.reshape(similarity(concatenation, pred), (-1, 1))
    return sim


def bilinear_merge_function(args, similarity):
    """
    Keras Merge function for the Bilinear Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    batch_size = pred.shape[0]
    n = subj.shape[1]

    rx = subj.reshape((batch_size, n, 1))
    rW = pred.reshape((batch_size, n, n))

    bilinear_transformation = (rx * rW).sum(1)

    sim = K.reshape(similarity(bilinear_transformation, obj), (-1, 1))
    return sim


def dual_bilinear_merge_function(args, similarity):
    """
    Keras Merge function for the Dual Bilinear Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    batch_size = pred.shape[0]
    n = subj.shape[1]

    pred_subj = pred[:, (n ** 2):]
    pred_obj = pred[:, :(n ** 2)]

    rx_subj = subj.reshape((batch_size, n, 1))
    rW_subj = pred_subj.reshape((batch_size, n, n))

    bilinear_transformation_subj = (rx_subj * rW_subj).sum(1)

    rx_obj = obj.reshape((batch_size, n, 1))
    rW_obj = pred_obj.reshape((batch_size, n, n))

    bilinear_transformation_obj = (rx_obj * rW_obj).sum(1)

    sim = K.reshape(similarity(bilinear_transformation_subj, bilinear_transformation_obj), (-1, 1))
    return sim


def affine_merge_function(args, similarity):
    """
    Keras Merge function for the Affine Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    batch_size = pred.shape[0]
    n = subj.shape[1]

    pred_W = pred[:, :(n ** 2)]
    pred_b = pred[:, (n ** 2):]

    rx = subj.reshape((batch_size, n, 1))
    rW = pred_W.reshape((batch_size, n, n))

    affine_transformation = (rx * rW).sum(1) + pred_b

    sim = K.reshape(similarity(affine_transformation, obj), (-1, 1))
    return sim


def dual_affine_merge_function(args, similarity):
    """
    Keras Merge function for the Dual Affine Embeddings model
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    batch_size = pred.shape[0]
    n = subj.shape[1]

    pred_subj = pred[:, ((n ** 2) + n):]
    pred_obj = pred[:, :((n ** 2) + n)]

    pred_W_subj = pred_subj[:, :(n ** 2)]
    pred_b_subj = pred_subj[:, (n ** 2):]

    pred_W_obj = pred_obj[:, :(n ** 2)]
    pred_b_obj = pred_obj[:, (n ** 2):]

    rx_subj = subj.reshape((batch_size, n, 1))
    rW_subj = pred_W_subj.reshape((batch_size, n, n))

    affine_transformation_subj = (rx_subj * rW_subj).sum(1) + pred_b_subj

    rx_obj = subj.reshape((batch_size, n, 1))
    rW_obj = pred_W_obj.reshape((batch_size, n, n))

    affine_transformation_obj = (rx_obj * rW_obj).sum(1) + pred_b_obj

    sim = K.reshape(similarity(affine_transformation_subj, affine_transformation_obj), (-1, 1))
    return sim


def manifold_sphere_merge_function(args, similarity):
    """
    Keras Merge function for the ManifoldE (Sphere) model described in:
        H Xiao et al. - From One Point to A Manifold: Knowledge Graph Embedding For Precise Link Prediction - IJCAI 2016
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    batch_size = pred.shape[0]
    n = subj.shape[1]

    translation = subj + pred[:, :n]
    M = - K.reshape(similarity(translation, obj), (-1, 1))
    D = pred[:, n:]

    return norms.square_l2(M - (D ** 2))


def manifold_hyperplane_merge_function(args, similarity):
    """
    Keras Merge function for the ManifoldE (Hyperplane) model described in:
        H Xiao et al. - From One Point to A Manifold: Knowledge Graph Embedding For Precise Link Prediction - IJCAI 2016
    :param args: List of two arguments: the former containing the relation embedding,
        and the latter containing the two entity embeddings.
    :param similarity: Similarity function.
    :return: The similarity between the translated subject embedding, and the object embedding.
    """
    subj, pred, obj = to_triples(args)

    batch_size = pred.shape[0]
    n = subj.shape[1]

    translation = pred[:, :(n * 2)]

    translation_subj = subj + translation[:, :n]
    translation_obj = obj + translation[:, n:]

    M = - K.reshape(similarity(translation_subj, translation_obj), (-1, 1))
    D = pred[:, (n * 2):]

    return norms.square_l2(M - (D ** 2))


# aliases
TransE = TranslatingEmbeddings = translating_merge_function
DualTransE = DualTranslatingEmbeddings = dual_translating_merge_function

ScalE = ScalEQ = ScalingEmbeddings = scaling_merge_function
DualScalE = DualScalingEmbeddings = dual_scaling_merge_function
ScalTransE = ScalingTranslatingEmbeddings = scaling_translating_merge_function

HolE = HolographicEmbeddings = holographic_merge_function

DAffinE = DiagonalAffineEmbeddings = diagonal_affine_merge_function
DualDAffinE = DualDiagonalAffineEmbeddings = dual_diagonal_affine_merge_function

ConcatE = ConcatenatingEmbeddings = concatenate_merge_function

BilinearE = RESCAL = BilinearEmbeddings = bilinear_merge_function
DualBilinearE = DualRESCAL = DualBilinearEmbeddings = dual_bilinear_merge_function

AffinE = AffineEmbeddings = affine_merge_function
DualAffinE = DualAffineEmbeddings = dual_affine_merge_function

ManifoldESphere = manifold_sphere_merge_function
ManifoldEHyperplane = manifold_hyperplane_merge_function

def get_function(function_name):
    this_module = sys.modules[__name__]
    if hasattr(this_module, function_name):
        function = getattr(this_module, function_name)
    else:
        raise ValueError("Unknown merge function: %s" % function_name)
    return function
