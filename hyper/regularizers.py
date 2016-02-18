# -*- coding: utf-8 -*-

from keras.regularizers import Regularizer
from keras import backend as K


class GroupRegularizer(Regularizer):
    def __init__(self, regularizers):
        self.regularizers = regularizers

    def set_param(self, p):
        for regularizer in self.regularizers:
            regularizer.set_param(p)

    def __call__(self, loss):
        for regularizer in self.regularizers:
            loss += regularizer(loss)
            return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class TranslationRuleRegularizer(Regularizer):
    def __init__(self, head, tail, l=0.01):
        self.head, self.tail, self.l = head, tail, l

    def set_param(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, loss):
        head_embedding = self.embeddings[self.head, :]
        tail_embedding = None

        for hop, is_reversed in self.tail:
            hop_embedding = (- 1.0 if is_reversed is True else 1.0) * self.embeddings[hop, :]
            tail_embedding = hop_embedding if tail_embedding is None else (tail_embedding + hop_embedding)

        diff = K.sum(K.abs(head_embedding - tail_embedding))

        loss += diff * self.l
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__, "l": self.l}


class ScalingRuleRegularizer(Regularizer):
    def __init__(self, head, tail, l=0.01):
        self.head, self.tail, self.l = head, tail, l

    def set_param(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, loss):
        head_embedding = self.embeddings[self.head, :]
        tail_embedding = None

        for hop, is_reversed in self.tail:
            hop_embedding = (1. / self.embeddings[hop, :] if is_reversed is True else self.embeddings[hop, :])
            tail_embedding = hop_embedding if tail_embedding is None else (tail_embedding * hop_embedding)

        diff = K.sum(K.abs(head_embedding - tail_embedding))

        loss += diff * self.l
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__, "l": self.l}
