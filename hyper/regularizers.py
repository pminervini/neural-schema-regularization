# -*- coding: utf-8 -*-

from keras.regularizers import Regularizer
from keras import backend as K


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

    def get_config(self):
        return {"name": self.__class__.__name__, "l": self.l}
