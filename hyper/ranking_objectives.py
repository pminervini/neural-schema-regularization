# -*- coding: utf-8 -*-

from keras import backend as K

from hyper import objectives


def margin_based_loss(y_true, y_pred, negatives_per_positive_example=2, *args, **kwargs):
    pos, diff, N = y_pred[0::3], .0, negatives_per_positive_example + 1
    for j in range(1, N):
        neg = y_pred[j::N]
        # loss = max{margin - 1 (pos - neg), 0}
        diff += objectives.hinge_loss(1, pos - neg, *args, **kwargs)
    target = y_true[0::N]
    return diff.sum() + target.sum()
