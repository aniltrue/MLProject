from TIMIT.DataReader import get_data
from RNNLayers.LSTM import LSTM
from RNNLayers.LSTMVariants import LSTMVariants
from RNNLayers.GRU import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
from RNNExperimentCallBack import RNNExperimentCallBack

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "TIMIT/output/")
EXPERIMENTS_FILE = os.path.join(ROOT_DIR, "experiments.csv")


def get_model(name: str, cell_name: str, layer_size: int = 32, output_size: int = 60, seq_length: int = 2040):
    model = Sequential(name=name)

    if cell_name == "LSTM":
        rnn = LSTM(layer_size)
    elif cell_name == "GRU":
        rnn = GRU(layer_size)
    elif cell_name == "NP":
        rnn = LSTM(layer_size, peephole=False)
    else:
        rnn = LSTMVariants(layer_size, cell_name)

    model.add(Input(shape=(seq_length, 1), dtype="float32", name="input"))
    model.add(rnn)
    model.add(Dense(output_size, activation="softmax"))

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["acc"])

    return model


def train_model(model, x_train, x_test, y_train, y_test, epochs: int = 2, batch_size: int = 64):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[RNNExperimentCallBack(model, "TIMIT", batch_size, OUTPUT_DIR, EXPERIMENTS_FILE)])

    del model


def train(x_train, x_test, y_train, y_test):
    # train_model(get_model("LSTM", "LSTM"), x_train, x_test, y_train, y_test)
    train_model(get_model("GRU", "GRU"), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NIG", "NIG"), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NFG", "NFG"), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NOG", "NOG"), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NP", "NP"), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NIAF", "NIAF"), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NOAF", "NOAF"), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_FGR", "FGR"), x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()

    train(x_train, x_test, y_train, y_test)