from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras import initializers
from abc import ABC, abstractmethod


class AbstractRNNCell(Layer, ABC):
    def __init__(self,
                 units: int,
                 kernel_activation: str = "tanh",
                 recurrent_activation: str = "hard_sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros",
                 use_bias: bool = True,
                 **kwargs):

        super(AbstractRNNCell, self).__init__(**kwargs)

        self.units = units
        self.kernel_activation = activations.get(kernel_activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

    @abstractmethod
    def call(self, inputs, states, **kwargs):
        raise NotImplementedError("Abstract Method")

    @abstractmethod
    def get_initial_state(self, inputs):
        raise NotImplementedError("Abstract Method")


class AbstractRNN(Layer, ABC):
    def __init__(self, units: int, cell, reversed: bool = False, return_sequences: bool = False, **kwargs):
        super(AbstractRNN, self).__init__(**kwargs)

        self.units = units
        self.cell = cell
        self.reversed = reversed
        self.return_sequences = return_sequences

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if isinstance(input_shape, tf.TensorShape):
            input_shape = tuple(input_shape.as_list())

        cell_input_shape = input_shape[1:]

        if isinstance(self.cell, Layer) and not self.cell.built:
            with K.name_scope(self.cell.name):
                self.cell.build(cell_input_shape)

        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)

        def step_fn(inputs, states):
            output, new_states = self.cell.call(inputs, states, **kwargs)

            return output, new_states

        initial_states = self.cell.get_initial_state(inputs)

        last_output, outputs, states = K.rnn(step_fn, inputs, initial_states=initial_states, go_backwards=self.reversed, input_length=input_shape[1])

        if self.return_sequences:
            return states
        else:
            return last_output


class AbstractBiRNN(Layer, ABC):
    def __init__(self, units: int, cell_f: AbstractRNNCell, cell_b: AbstractRNNCell, return_sequences: bool = False, **kwargs):
        super(AbstractBiRNN, self).__init__(**kwargs)

        self.units = units
        self.cell_f = cell_f
        self.cell_b = cell_b
        self.reversed = reversed
        self.return_sequences = return_sequences

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

        initial_states_f = self.cell_f.get_initial_state(inputs)
        initial_states_b = self.cell_b.get_initial_state(inputs)

        last_output_f, outputs_f, states_f = K.rnn(step_f, inputs, initial_states=initial_states_f, go_backwards=False, input_length=input_shape[1])
        last_output_b, outputs_b, states_b = K.rnn(step_b, inputs, initial_states=initial_states_b, go_backwards=True, input_length=input_shape[1])

        last_output = K.concatenate([last_output_f, last_output_b])
        outputs = K.concatenate([outputs_f, outputs_b])

        if self.return_sequences:
            return outputs
        else:
            return last_output