# -*- coding: utf-8 -*-

from keras.layers.core import Layer

import sys


class Lambda(Layer):
    def __init__(self, function, output_shape=None, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        py3 = sys.version_info[0] == 3
        self.function = function
        if output_shape is None:
            self._output_shape = None
        elif type(output_shape) in {tuple, list}:
            self._output_shape = tuple(output_shape)
        else:
            self._output_shape = output_shape
        super(Lambda, self).__init__()

    @property
    def output_shape(self):
        if self._output_shape is None:
            return self.input_shape
        elif type(self._output_shape) == tuple:
            return (self.input_shape[0], ) + self._output_shape
        else:
            output_shape_func = self._output_shape
            shape = output_shape_func(self.input_shape)
            if type(shape) not in {list, tuple}:
                raise Exception('output_shape function must return a tuple')
            return tuple(shape)

    def get_output(self, train=False):
        X = self.get_input(train)
        func = self.function
        return func(X)


class LambdaMerge(Lambda):
    def __init__(self, layers, function, output_shape=None):
        if len(layers) < 2:
            raise Exception('Please specify two or more input layers '
                            '(or containers) to merge.')
        self.layers = layers
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        for l in self.layers:
            params, regs, consts, updates = l.get_params()
            self.regularizers += regs
            self.updates += updates
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)
        py3 = sys.version_info[0] == 3
        self.function = function
        if output_shape is None:
            self._output_shape = None
        elif type(output_shape) in {tuple, list}:
            self._output_shape = tuple(output_shape)
        else:
            self._output_shape = output_shape
        super(Lambda, self).__init__()

    @property
    def output_shape(self):
        input_shapes = [layer.output_shape for layer in self.layers]
        if self._output_shape is None:
            return input_shapes[0]
        elif type(self._output_shape) == tuple:
            return (input_shapes[0][0], ) + self._output_shape
        else:
            output_shape_func = self._output_shape
            shape = output_shape_func(input_shapes)
            if type(shape) not in {list, tuple}:
                raise Exception('output_shape function must return a tuple.')
            return tuple(shape)

    def get_params(self):
        return self.params, self.regularizers, self.constraints, self.updates

    def get_output(self, train=False):
        func = self.function
        inputs = [layer.get_output(train) for layer in self.layers]
        return func(inputs)

    def get_input(self, train=False):
        res = []
        for i in range(len(self.layers)):
            o = self.layers[i].get_input(train)
            if not type(o) == list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res

    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = []
        for l in self.layers:
            weights += l.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].params)
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layers': [l.get_config() for l in self.layers],
                  'function': self.function,
                  'output_shape': self._output_shape}
        base_config = super(LambdaMerge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
