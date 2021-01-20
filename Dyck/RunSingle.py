from RNNLayers.LSTM import LSTM, BiLSTM
from RNNLayers.LSTMVariants import LSTMVariants, BiLSTMVariants
from RNNLayers.GRU import GRU, BiGRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from Dyck.Train import get_model
from RNNLayers.RNNBuilder import RNNCellBuilder
from RNNLayers.AbstractRNN import AbstractRNN
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

LETTERS = ["(", "[", "{", ")", "]", "}"]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/")
EXPERIMENTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments.csv")


class StatefulRNN(AbstractRNN):
    def __init__(self, units: int, cell: RNNCellBuilder, gate: str = "", **kwargs):
        super(StatefulRNN, self).__init__(units, cell, **kwargs)

        self.sequence_values = []
        self.gate = gate

        if gate != "":
            self.cell.return_gate = True
            self.cell.gate_name = gate
            self.cell.states_number = len(self.cell.states) + 1

    def call(self, inputs, **kwargs):
        self.sequence_values = []
        input_shape = K.int_shape(inputs)

        def step_fn(inputs, states):
            output, new_states = self.cell.call(inputs, states, **kwargs)

            return output, new_states

        initial_states = self.cell.get_initial_state(inputs)

        last_output, outputs, states = K.rnn(step_fn, inputs, initial_states=initial_states, go_backwards=self.reversed,
                                             input_length=input_shape[1])

        return outputs


def generate_input(s: str, par_size: int, max_seq: int = 20):
    inp = np.zeros((1, max_seq, par_size))

    start_index = max_seq - len(s)

    for i, p in enumerate(s):
        index = LETTERS.index(p)

        value = index % 3

        identity = np.identity(par_size)

        multiplier = 1 if index < 3 else -1

        inp[0, start_index + i, :] = multiplier * identity[value, :]

    return inp


def get_model_path(name: str, par_size: int):
    dataset_name = "Dyck%d" % par_size if par_size > 1 else "Dyck"

    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.startswith("%s_%s_" % (name, dataset_name)) and file.endswith(".h5"):
                return root + str(file)

    return "" # Error


def load_model(name: str, cell_name: str, par_size: int) -> RNNCellBuilder:
    model = get_model(name, cell_name, par_size=par_size)

    weights_path = get_model_path(name, par_size)

    model.load_weights(weights_path)

    model.summary()

    return model.layers[1].cell


def get_states(cell: RNNCellBuilder, par_size: int, x: np.ndarray):
    input_layer = Input(shape=(20, par_size), name="input")

    rnn = StatefulRNN(20, cell, return_sequences=True)
    out = rnn(input_layer)

    model = Model(inputs=[input_layer], outputs=[out])
    model.compile(optimizer=Adam(), loss="mse")

    outputs = model.predict(x)[0]

    return outputs


def get_gate(cell: RNNCellBuilder, par_size: int, x: np.ndarray, gate: str):
    input_layer = Input(shape=(20, par_size), name="input")

    rnn = StatefulRNN(20, cell, return_sequences=True, gate=gate)
    out = rnn(input_layer)

    model = Model(inputs=[input_layer], outputs=[out])
    model.compile(optimizer=Adam(), loss="mse")

    outputs = model.predict(x)[0]

    return outputs


def draw(s: str, outputs: np.ndarray):
    ticks = s

    while len(ticks) < 20:
        ticks = " " + ticks

    plt.plot(list(range(20)), outputs)
    plt.xticks(list(range(20)), ticks)
    plt.show()


if __name__ == "__main__":
    cell = load_model("LSTM_NFG", "NFG", 1)
    print("Cell Name:", cell.name)

    s = input("Enter Sequence:")

    x = generate_input(s, par_size=1)

    states = get_states(cell, 1, x)
    input_gate = get_gate(cell, 1, x, "input")
    # forget_gate = get_gate(cell, 1, x, "forget")
    output_gate = get_gate(cell, 1, x, "output")

    print(states.shape)
    # print(forget_gate.shape)
    print(input_gate.shape)
    print(output_gate.shape)

    # print(forget_gate[18, :])

    draw(s, input_gate)
    draw(s, output_gate)
