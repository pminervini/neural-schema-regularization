# -*- coding: utf-8 -*-

import abc

from keras.regularizers import Regularizer
from keras import backend as K

from hyper import similarities


class GroupRegularizer(Regularizer):
    def __init__(self, regularizers, uses_learning_phase=True):
        self.regularizers = regularizers
        self.uses_learning_phase = uses_learning_phase

    def set_param(self, p):
        for regularizer in self.regularizers:
            regularizer.set_param(p)

    def __call__(self, loss):
        for regularizer in self.regularizers:
            loss += regularizer(loss)
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class RuleRegularizer(Regularizer):
    def __init__(self, similarity=similarities.l2sqr, l=0.):
        self.similarity = similarity
        self.l = K.cast_to_floatx(l)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    @abc.abstractmethod
    def __call__(self, loss):
        while False:
            yield None

    def get_config(self):
        return {"similarity": self.similarity.__name__, "l": self.l}


class TranslationRuleRegularizer(RuleRegularizer):
    def __init__(self, head, tail, *args, **kwargs):
        super(TranslationRuleRegularizer, self).__init__(*args, **kwargs)
        self.head, self.tail = head, tail

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on RuleRegularizer instance before calling the instance. ')

        head_embedding = self.p[self.head, :]
        tail_embedding = None

        for hop, is_reversed in self.tail:
            hop_embedding = (- 1.0 if is_reversed is True else 1.0) * self.p[hop, :]
            tail_embedding = hop_embedding if tail_embedding is None else (tail_embedding + hop_embedding)

        sim = K.reshape(self.similarity(head_embedding, tail_embedding, axis=-1), (1,))[0]

        regularized_loss = loss - sim * self.l
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        sc = super(TranslationRuleRegularizer, self).get_config()
        config = {"name": self.__class__.__name__}
        config.update(sc)
        return config


class ScalingRuleRegularizer(RuleRegularizer):
    def __init__(self, head, tail, *args, **kwargs):
        super(ScalingRuleRegularizer, self).__init__(*args, **kwargs)
        self.head, self.tail = head, tail

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on RuleRegularizer instance before calling the instance. ')

        head_embedding = self.p[self.head, :]
        tail_embedding = None

        for hop, is_reversed in self.tail:
            hop_embedding = (1. / self.p[hop, :] if is_reversed is True else self.p[hop, :])
            tail_embedding = hop_embedding if tail_embedding is None else (tail_embedding * hop_embedding)

        sim = K.reshape(self.similarity(head_embedding, tail_embedding, axis=-1), (1,))[0]

        regularized_loss = loss - sim * self.l
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        sc = super(ScalingRuleRegularizer, self).get_config()
        config = {"name": self.__class__.__name__}
        config.update(sc)
        return config
