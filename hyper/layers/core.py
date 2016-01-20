# -*- coding: utf-8 -*-

from keras.layers.core import Layer


class Arguments(Layer):

    def __init__(self, arguments, **kwargs):
        super(Arguments, self).__init__(**kwargs)
        self.arguments = arguments
        self._input_shape = (None,)

    def get_output(self, train=False):
        return self.arguments

    def get_input(self, train=False):
        return None
