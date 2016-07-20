# -*- coding: utf-8 -*-

import abc

from keras.regularizers import Regularizer
from keras import backend as K

from hyper import similarities


class L1(Regularizer):
    def __init__(self, l1=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` before calling the instance.')
        regularized_loss = loss + K.sum(K.abs(self.p)) * self.l1
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__, 'l1': self.l1}


class L2(Regularizer):
    def __init__(self, l2=0.):
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` before calling the instance.')
        regularized_loss = loss + K.sum(K.square(self.p)) * self.l2
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__, 'l2': self.l2}


class ActivityL1(Regularizer):
    def __init__(self, l1=0., axis=0):
        self.l1 = K.cast_to_floatx(l1)
        self.axis = axis
        self.uses_learning_phase = True

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` before calling the instance.')
        regularized_loss = loss
        for i in range(len(self.layer.inbound_nodes)):
            output = self.layer.get_output_at(i)
            regularized_loss += self.l1 * K.sum(K.mean(K.abs(output), axis=self.axis))
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__, 'l1': self.l1}


class ActivityL2(Regularizer):
    def __init__(self, l2=0., axis=0):
        self.l2 = K.cast_to_floatx(l2)
        self.axis = axis
        self.uses_learning_phase = True

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` before calling the instance.')
        regularized_loss = loss
        for i in range(len(self.layer.inbound_nodes)):
            output = self.layer.get_output_at(i)
            regularized_loss += self.l2 * K.sum(K.mean(K.square(output), axis=self.axis))
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__, 'l2': self.l2}


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
    def __init__(self, similarity=similarities.l2sqr, l=0., entity_embedding_size=None, *args, **kwargs):
        self.similarity = similarity
        self.l = K.cast_to_floatx(l)
        self.entity_embedding_size = entity_embedding_size
        self.uses_learning_phase = True

    @property
    def entity_embedding_size(self):
        return self._entity_embedding_size

    @entity_embedding_size.setter
    def entity_embedding_size(self, entity_embedding_size):
        self._entity_embedding_size = entity_embedding_size

    def set_param(self, p):
        self.p = p

    @abc.abstractmethod
    def __call__(self, loss):
        while False:
            yield None

    def get_config(self):
        return {"similarity": self.similarity.__name__,
                "entity_embedding_size": self.entity_embedding_size,
                "l": self.l}


class TranslationRuleRegularizer(RuleRegularizer):
    def __init__(self, head, tail, *args, **kwargs):
        super(TranslationRuleRegularizer, self).__init__(*args, **kwargs)
        self.head, self.tail = head, tail

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on RuleRegularizer instance before calling the instance.')

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


DualTranslationRuleRegularizer = TranslationRuleRegularizer


class ScalingRuleRegularizer(RuleRegularizer):
    def __init__(self, head, tail, *args, **kwargs):
        super(ScalingRuleRegularizer, self).__init__(*args, **kwargs)
        self.head, self.tail = head, tail

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on RuleRegularizer instance before calling the instance.')

        head_embedding = self.p[self.head, :]
        tail_embedding = None

        for hop, is_reversed in self.tail:
            hop_embedding = (1. / self.p[hop, :]) if is_reversed is True else self.p[hop, :]
            tail_embedding = hop_embedding if tail_embedding is None else (tail_embedding * hop_embedding)

        sim = K.reshape(self.similarity(head_embedding, tail_embedding, axis=-1), (1,))[0]

        regularized_loss = loss - sim * self.l
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        sc = super(ScalingRuleRegularizer, self).get_config()
        config = {"name": self.__class__.__name__}
        config.update(sc)
        return config


class DivMultRuleRegularizer(RuleRegularizer):
    def __init__(self, head, tail, *args, **kwargs):
        super(DivMultRuleRegularizer, self).__init__(*args, **kwargs)
        self.head, self.tail = head, tail

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on RuleRegularizer instance before calling the instance.')

        head_embedding = self.p[self.head, :]
        tail_embedding = None

        for hop, is_reversed in self.tail:
            # hop_embedding = (1. / self.p[hop, :]) if is_reversed is True else self.p[hop, :]
            hop_embedding = self.p[hop, :]
            tail_embedding = hop_embedding if tail_embedding is None else (tail_embedding * hop_embedding)

        sim = K.reshape(self.similarity(head_embedding, tail_embedding, axis=-1), (1,))[0]

        regularized_loss = loss - sim * self.l
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        sc = super(DivMultRuleRegularizer, self).get_config()
        config = {"name": self.__class__.__name__}
        config.update(sc)
        return config
