import tensorflow as tf
from RNNLayers.AbstractRNN import AbstractRNNCell, AbstractRNN
from tensorflow.keras.layers import Layer
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


class BiLSTM(Layer):
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

        self.cell_f = LSTMCell(units, activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer, **kwargs)

        self.cell_b = LSTMCell(units, activation, recurrent_activation,
                        kernel_initializer, recurrent_initializer, bias_initializer, **kwargs)

        super(BiLSTM, self).__init__(**kwargs)

    def get_initial_state(self, inputs):
        batch_size = tf.compat.v1.shape(inputs)[0]
        dtype = inputs.dtype

        flat_dims = tf.TensorShape([self.units]).as_list()
        init_state_size = [batch_size] + flat_dims

        return tf.zeros(init_state_size, dtype=dtype), tf.zeros(init_state_size, dtype=dtype)

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