from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K
from RNNLayers.RNNCellBuilder import RNNCellBuilder
from tensorflow.keras import activations
from tensorflow.keras import initializers


class NewGRU(Layer):
    def __init__(self, units: int, reversed: bool = False, return_sequences: bool = False, **kwargs):
        super(NewGRU, self).__init__(**kwargs)

        cell = RNNCellBuilder(units, states=["h"])

        cell.add_recurrent("update", ["X", "h"])
        cell.add_recurrent("reset", ["X", "h"])
        cell.add_kernel("output", ["X", "reset"])
        cell.add_var("h_next", ["update", "h", "output"], lambda X: (1 - X[0]) * X[1] + X[0] * X[2])

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

        initial_states = self.cell.get_initial_state(inputs)

        last_output, outputs, states = K.rnn(step_fn, inputs, initial_states=initial_states, go_backwards=self.reversed, input_length=input_shape[1])

        if self.return_sequences:
            return outputs
        else:
            return last_output