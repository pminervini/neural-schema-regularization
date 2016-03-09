# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error:

    .. math:: L = (p - t)^2

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Error value
    """
    return K.mean(K.square(y_pred - y_true), axis=-1)


def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error:

    .. math:: L = \\sqrt{(p - t)^2}


    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Error value
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error:

    .. math:: L = \\abs{p - t}

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Error value
    """
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred, eps=1e-6):
    """
    Men Absolute Percentage Error:

    .. math:: L = \\frac{\\abs{p - t}}{\\abs{t}}

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param eps: Epsilon
    :return: Error value
    """
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), eps, np.inf))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred, eps=1e-6):
    """
    Men Squared Logarithmic Error:

    .. math:: L = (\\log(p + 1) - \\log(t + 1))^2

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param eps: Epsilon
    :return: Error value
    """
    first_log = K.log(K.clip(y_pred, eps, np.inf) + 1.)
    second_log = K.log(K.clip(y_true, eps, np.inf) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge_loss(y_true, y_pred, margin=1.):
    """
    Squared Hinge Loss:

    .. math:: L = \\max(\\lambda - t * p, 0)^2

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param margin: Margin
    :return: Loss value
    """
    return K.mean(K.square(K.maximum(margin - y_true * y_pred, 0.)), axis=-1)


def hinge_loss(y_true, y_pred, margin=1.):
    """
    Hinge Loss:

    .. math:: L = \\max(\\lambda - t * p, 0)

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param margin: Margin
    :return: Loss value
    """
    return K.mean(K.maximum(margin - y_true * y_pred, 0.), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    """
    Categorical Cross-Entropy.

    The cross entropy between two probability distributions measures the
    average number of bits needed to identify an event from a set of
    possibilities, if a coding scheme is used based on a given probability
    distribution, rather than the “true” distribution.

    Expects a binary class matrix instead of a vector of scalar classes.

    .. math:: L_i = - \\sum_j t_{i, j} \\log(p_{i, j})

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Loss value
    """
    return K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)


def binary_crossentropy(y_true, y_pred):
    """
    Binary Cross-Entropy.

    .. math:: L = -t \\log(p) - (1 - t) \\log(1 - p)

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Loss value
    """
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def poisson_loss(y_true, y_pred, eps=1e-6):
    """
    Poisson Loss.
        avg(p - t * log(p))

    .. math:: L = p - t \\log(p)

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param eps: Epsilon
    :return: Loss value
    """
    return K.mean(y_pred - y_true * K.log(y_pred + eps), axis=-1)


def logistic_loss(y_true, y_pred):
    """
    Logistic Loss.
        avg(log(1 + exp(- yp)))

    .. math:: L = \\log(1 + \\exp(- yp))

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Loss value
    """
    return K.mean(K.log(1. + K.exp(- y_true * y_pred)), axis=-1)


# aliases
mse = MSE = mean_squared_error
rmse = RMSE = root_mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error

shl = SHL = squared_hinge_loss
hl = HL = hinge_loss
cc = CC = categorical_crossentropy
bc = BC = binary_crossentropy
pl = PL = poisson_loss
