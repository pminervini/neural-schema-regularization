# -*- coding: utf-8 -*-

from keras import backend as K


def negative_l1_distance(x, y, axis=1):
    """
    Negative L1 Distance.

    .. math:: L = - \\sum_i \\abs(x_i - y_i)

    :param x: First term.
    :param y: Second term.
    :param axis: Axis.
    :return: Similarity Value.
    """
    return - K.sum(K.abs(x - y), axis=axis)


def negative_l2_distance(x, y, axis=1):
    """
    Negative L2 Distance.

    .. math:: L = - \\sqrt{\\sum_i (x_i - y_i)^2}

    :param x: First term.
    :param y: Second term.
    :param axis: Axis.
    :return: Similarity Value.
    """
    return - K.sqrt(K.sum(K.square(x - y), axis=axis))


def negative_square_l2_distance(x, y, axis=1):
    """
    Negative Square L2 Distance.

    .. math:: L = - \\sum_i (x_i - y_i)^2

    :param x: First term.
    :param y: Second term.
    :param axis: Axis.
    :return: Similarity Value.
    """
    return - K.sum(K.square(x - y), axis=axis)


def dot_product(x, y, axis=1):
    """
    Dot Product.

    .. math:: L = \\sum_i x_i y_i

    :param x: First term.
    :param y: Second term.
    :param axis: Axis.
    :return: Similarity Value.
    """
    return K.sum(x * y, axis=axis)


def cosine_similarity(x, y, axis=1):
    """
    Cosine Similarity

    .. math:: L = \\frac{\\sum_i x_i y_i}{\\norm{x} \\norm{y}}

    :param x: First term.
    :param y: Second term.
    :param axis: Axis.
    :return: Similarity Value.
    """
    x_norm = K.l2_normalize(x, axis=axis)
    y_norm = K.l2_normalize(y, axis=axis)
    return K.sum(x_norm * y_norm, axis=axis)


# aliases
l1 = L1 = negative_l1_distance
l2 = L2 = negative_l2_distance
l2sqr = L2SQR = negative_square_l2_distance
dot = DOT = dot_product
cosine = COSINE = cosine_similarity
