# -*- coding: utf-8 -*-

from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam, Adamax

import inspect
import logging


def make_optimizer(optimizer_name, lr=0.01, momentum=0., decay=0., nesterov=False, epsilon=1e-6, rho=0.95,
                  beta_1=0.9, beta_2=0.999):
    """
    Returns a Keras Optimizer.
    :param optimizer_name: Name of the optimizer - sgd, adagrad, adadelta, rmsprop, adam.
    :param lr: learning rate.
    :param momentum: momentum (used by sgd).
    :param decay: decay (used by sgd).
    :param nesterov: whether to use the Nesterov Accelerated Gradient SGD variant.
    :param epsilon: epsilon (used by adagrad, adadelta, rmsprop, adam and adamax).
    :param rho: rho (used by adadelta, rmsprop).
    :param beta_1: beta_1 (used by adam and adamax).
    :param beta_2: beta_2 (used by adam and adamax).
    :return: a Keras Optimizer.
    """

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    logging.info('Optimizer: %s' % {arg: values[arg] for arg in args if len(str(values[arg])) < 32})

    optimizer = None

    if optimizer_name == 'sgd':
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
    elif optimizer_name == 'adagrad':
        optimizer = Adagrad(lr=lr, epsilon=epsilon)
    elif optimizer_name == 'adadelta':
        optimizer = Adadelta(lr=lr, rho=rho, epsilon=epsilon)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon)
    elif optimizer_name == 'adam':
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif optimizer_name == 'adamax':
        optimizer = Adamax(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    if optimizer is None:
        raise ValueError('Unknown optimizer: %s' % optimizer_name)

    return optimizer
