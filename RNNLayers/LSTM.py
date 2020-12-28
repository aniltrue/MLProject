import tensorflow as tf
from RNNLayers.AbstractRNN import AbstractRNNCell, AbstractRNN
import tensorflow.keras.backend as K


class LSTMCell(AbstractRNNCell):
    def __init__(self,
                 units: int,
                 activation: str = "tanh",
                 recurrent_activation: str = "sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros", **kwargs):

        super(LSTMCell, self).__init__(units, activation, recurrent_activation,
                                       kernel_initializer, recurrent_initializer, bias_initializer, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Input Gate
        self.ki = self.add_weight("kernel_input", (input_dim, self.units), initializer=self.kernel_initializer)
        self.ri = self.add_weight("recurrent_input", (self.units, self.units), initializer=self.recurrent_initializer)
        self.bi = self.add_weight("bias_input", (self.units, ), initializer=self.bias_initializer)

        # Forget Gate
        self.kf = self.add_weight("kernel_forget", (input_dim, self.units), initializer=self.kernel_initializer)
        self.rf = self.add_weight("recurrent_forget", (self.units, self.units), initializer=self.recurrent_initializer)
        self.bf = self.add_weight("bias_forget", (self.units,), initializer=self.bias_initializer)

        # Output Gate
        self.ko = self.add_weight("kernel_output", (input_dim, self.units), initializer=self.kernel_initializer)
        self.ro = self.add_weight("recurrent_output", (self.units, self.units), initializer=self.recurrent_initializer)
        self.bo = self.add_weight("bias_output", (self.units,), initializer=self.bias_initializer)

        # Cell Gate
        self.kc = self.add_weight("kernel_cell", (input_dim, self.units), initializer=self.kernel_initializer)
        self.rc = self.add_weight("recurrent_cell", (self.units, self.units), initializer=self.recurrent_initializer)
        self.bc = self.add_weight("bias_cell", (self.units,), initializer=self.bias_initializer)

    def call(self, inputs, states, **kwargs):
        h_prev = states[0][0]
        c_prev = states[0][1]

        _input = self.recurrent_activation(K.dot(inputs, self.ki) + K.dot(h_prev, self.ri))
        _forget = self.recurrent_activation(K.dot(inputs, self.kf) + K.dot(h_prev, self.rf))
        _output = self.recurrent_activation(K.dot(inputs, self.ko) + K.dot(h_prev, self.ro))

        _c = self.activation(K.dot(inputs, self.kc) + K.dot(h_prev, self.rc) + self.bc)

        c = self.recurrent_activation(_forget * c_prev + _input * _c)
        h = self.activation(c) * _output

        return h, [h, c]


class LSTM(AbstractRNN):
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

        cell = LSTMCell(units, activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer, **kwargs)

        super(LSTM, self).__init__(units, cell, reversed, return_sequences, **kwargs)

    def get_initial_state(self, inputs):
        batch_size = tf.compat.v1.shape(inputs)[0]
        dtype = inputs.dtype

        flat_dims = tf.TensorShape([self.units]).as_list()
        init_state_size = [batch_size] + flat_dims

        return tf.zeros(init_state_size, dtype=dtype), tf.zeros(init_state_size, dtype=dtype)