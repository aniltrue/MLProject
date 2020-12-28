import tensorflow as tf
from RNNLayers.AbstractRNN import AbstractRNNCell, AbstractRNN
import tensorflow.keras.backend as K


class GRUCell(AbstractRNNCell):
    def __init__(self,
                 units: int,
                 activation: str = "tanh",
                 recurrent_activation: str = "sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros", **kwargs):

        super(GRUCell, self).__init__(units, activation, recurrent_activation,
                                       kernel_initializer, recurrent_initializer, bias_initializer, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Update Gate
        self.ku = self.add_weight("kernel_update", (input_dim, self.units), initializer=self.kernel_initializer)
        self.ru = self.add_weight("recurrent_update", (self.units, self.units), initializer=self.recurrent_initializer)
        self.bu = self.add_weight("bias_update", (self.units, ), initializer=self.bias_initializer)

        # Reset Gate
        self.kr = self.add_weight("kernel_reset", (input_dim, self.units), initializer=self.kernel_initializer)
        self.rr = self.add_weight("recurrent_reset", (self.units, self.units), initializer=self.recurrent_initializer)
        self.br = self.add_weight("bias_reset", (self.units, ), initializer=self.bias_initializer)

        # Output Gate
        self.ko = self.add_weight("kernel_input", (input_dim, self.units), initializer=self.kernel_initializer)
        self.ro = self.add_weight("recurrent_input", (self.units, self.units), initializer=self.recurrent_initializer)
        self.bo = self.add_weight("bias_input", (self.units, ), initializer=self.bias_initializer)

    def call(self, inputs, states, **kwargs):
        h_prev = states[0]

        _update = self.recurrent_activation(K.dot(inputs, self.ku) + K.dot(h_prev, self.ru) + self.bu)
        _reset = self.recurrent_activation(K.dot(inputs, self.kr) + K.dot(h_prev, self.rr) + self.br)

        _h = self.activation(K.dot(inputs, self.ko) + K.dot(_reset * h_prev, self.ro) + self.bo)

        h = (1 - _update) * h_prev + _update * h_prev

        return h, h


class GRU(AbstractRNN):
    def __init__(self,
                 units: int,
                 activation: str = "tanh",
                 recurrent_activation: str = "sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 reversed: bool = False,
                 return_sequences: bool = False,
                 **kwargs):

        cell = GRUCell(units, activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer, **kwargs)

        super(GRU, self).__init__(units, cell, reversed, return_sequences, **kwargs)

    def get_initial_state(self, inputs):
        batch_size = tf.compat.v1.shape(inputs)[0]
        dtype = inputs.dtype

        flat_dims = tf.TensorShape([self.units]).as_list()
        init_state_size = [batch_size] + flat_dims

        return tf.zeros(init_state_size, dtype=dtype)