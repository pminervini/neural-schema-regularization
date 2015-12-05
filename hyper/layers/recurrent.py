# -*- coding: utf-8 -*-

from keras import backend as K
from keras import activations
from keras.layers.recurrent import Recurrent


class RecurrentTransE(Recurrent):
    '''
    Fully-connected Low-Capacity RNN where the output is to fed back to input.
    Takes inputs with shape (nb_samples, max_sample_length, input_dim), samples shorter than `max_sample_length` are
    padded with zeros at the end) and returns outputs with shape:
        if not return_sequences:    (nb_samples, output_dim)
        if return_sequences:        (nb_samples, max_sample_length, output_dim)
    '''
    def __init__(self, activation='linear', **kwargs):
        self.output_dim, self.states = None, None
        self.activation = activations.get(activation)
        super(RecurrentTransE, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            if not input_shape[0]:
                raise Exception('If a Recurrent Neural Network is stateful, a complete input_shape must be provided'
                                ' (including batch size).')
            self.states = [K.ones(input_shape[0], input_shape[2])]
        else:
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.params = []

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        assert len(states) == 1
        prev_output = states[0]
        output = self.activation(x + prev_output)
        return output, [output]

    def get_config(self):
        config = dict(activation=self.activation.__name__)
        base_config = super(RecurrentTransE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
