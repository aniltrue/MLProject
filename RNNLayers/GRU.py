import tensorflow as tf
from RNNLayers.AbstractRNN import AbstractRNNCell, AbstractRNN
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


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

        h = (1 - _update) * h_prev + _update * _h

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


class BiGRU(Layer):
    def __init__(self,
                 units: int,
                 activation: str = "tanh",
                 recurrent_activation: str = "sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 return_sequences: bool = False,
                 **kwargs):

        self.units = units
        self.return_sequences = return_sequences

        self.cell_f = GRUCell(units, activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer, **kwargs)

        self.cell_b = GRUCell(units, activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer, **kwargs)

        super(BiGRU, self).__init__(**kwargs)

    def get_initial_state(self, inputs):
        batch_size = tf.compat.v1.shape(inputs)[0]
        dtype = inputs.dtype

        flat_dims = tf.TensorShape([self.units]).as_list()
        init_state_size = [batch_size] + flat_dims

        return tf.zeros(init_state_size, dtype=dtype)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if isinstance(input_shape, tf.TensorShape):
            input_shape = tuple(input_shape.as_list())

        cell_input_shape = input_shape[1:]

        if isinstance(self.cell_f, Layer) and not self.cell_f.built:
            with K.name_scope(self.cell_f.name):
                self.cell_f.build(cell_input_shape)
                self.cell_f.built = True

        if isinstance(self.cell_b, Layer) and not self.cell_b.built:
            with K.name_scope(self.cell_b.name):
                self.cell_b.build(cell_input_shape)
                self.cell_b.built = True

        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)

        def step_f(inputs, states):
            output, new_states = self.cell_f.call(inputs, states, **kwargs)

            return output, new_states

        def step_b(inputs, states):
            output, new_states = self.cell_b.call(inputs, states, **kwargs)

            return output, new_states

        initial_states_f = self.get_initial_state(inputs),
        initial_states_b = self.get_initial_state(inputs),

        last_output_f, outputs_f, states_f = K.rnn(step_f, inputs, initial_states=initial_states_f, go_backwards=False, input_length=input_shape[1])
        last_output_b, outputs_b, states_b = K.rnn(step_b, inputs, initial_states=initial_states_b, go_backwards=True, input_length=input_shape[1])

        last_output = K.concatenate([last_output_f, last_output_b])
        outputs = K.concatenate([outputs_f, outputs_b])

        if self.return_sequences:
            return outputs
        else:
            return last_output