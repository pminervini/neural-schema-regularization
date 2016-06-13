# -*- coding: utf-8 -*-

from keras import backend as K
from hyper import objectives


def margin_based_loss(y_true, y_pred, nb_sample_sets=3, *args, **kwargs):
    """
    Margin-based Ranking Loss.
        max(0, margin - positive_score + negative_score)

    .. math:: L = \\max(0, \\lambda + n - p)

    :param y_true: Vector of zeros.
    :param y_pred: Vector of predictions, where scores of positive and negative examples are interleaved.
    :param nb_sample_sets: Number of negative examples following each positive example.
    :param args: Various.
    :param kwargs: Various.
    :return: Loss.
    """
    positive_scores, target, loss = y_pred[0::nb_sample_sets], y_true[0::nb_sample_sets], .0

    for j in range(1, nb_sample_sets):
        negative_scores = y_pred[j::nb_sample_sets]

        # loss = max{margin - 1 (positive_scores - negative_scores), 0}
        diff = positive_scores - negative_scores
        loss += K.sum(objectives.hinge_loss(1, diff, *args, **kwargs))

    return loss + K.sum(target)


def logistic_loss(y_true, y_pred, nb_sample_sets=3, *args, **kwargs):
    """
    Logistic Ranking Loss.
        log(1 + exp(positive_score - negative_score))

    .. math:: L = \\log(1 + \\exp(n - p))

    :param y_true: Vector of zeros.
    :param y_pred: Vector of predictions, where scores of positive and negative examples are interleaved.
    :param nb_sample_sets: Number of negative examples following each positive example.
    :param args: Various.
    :param kwargs: Various.
    :return: Loss.
    """
    positive_scores, target, loss = y_pred[0::nb_sample_sets], y_true[0::nb_sample_sets], .0

    for j in range(1, nb_sample_sets):
        negative_scores = y_pred[j::nb_sample_sets]

        # loss = log(1 + exp(- (positive_scores - negative_scores)))
        diff = positive_scores - negative_scores
        loss += K.sum(objectives.logistic_loss(1, diff))

    return loss + K.sum(target)


# aliases
hinge = margin_based_loss
logistic = logistic_loss
