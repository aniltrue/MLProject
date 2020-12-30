from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras import initializers
from abc import ABC, abstractmethod


class AbstractRNNCell(Layer, ABC):
    def __init__(self,
                 units: int,
                 activation: str = "tanh",
                 recurrent_activation: str = "sigmoid",
                 kernel_initializer: str = "glorot_uniform",
                 recurrent_initializer: str = "orthogonal",
                 bias_initializer: str = "zeros", **kwargs):

        super(AbstractRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    @abstractmethod
    def call(self, inputs, states, **kwargs):
        raise NotImplementedError("Abstract Method")

    @abstractmethod
    def get_initial_state(self, inputs):
        pass


class AbstractRNN(Layer, ABC):
    def __init__(self, units: int, cell: AbstractRNNCell, reversed: bool = False, return_sequences: bool = False, **kwargs):
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
                self.cell.built = True

        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)

        def step_fn(inputs, states):
            output, new_states = self.cell.call(inputs, states, **kwargs)

            return output, new_states

        initial_states = self.cell.get_initial_state(inputs),

        last_output, outputs, states = K.rnn(step_fn, inputs, initial_states=initial_states, go_backwards=self.reversed, input_length=input_shape[1])

        if self.return_sequences:
            return outputs
        else:
            return last_output
