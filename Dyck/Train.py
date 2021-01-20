from Dyck.DataGenerator import get_data
from RNNLayers.LSTM import LSTM, BiLSTM
from RNNLayers.LSTMVariants import LSTMVariants, BiLSTMVariants
from RNNLayers.GRU import GRU, BiGRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os
from RNNExperimentCallBack import RNNExperimentCallBack
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/")
EXPERIMENTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments.csv")

EPOCH = 1000
BATCH_SIZE = 512
MIN_SEQUENCE = 4
MAX_SEQUENCE = 20
HIDDEN_UNITS = 20
PAR_SIZE = 3


def get_model(name: str, cell_name: str, par_size: int = PAR_SIZE, max_seq: int = MAX_SEQUENCE, layer_size: int = HIDDEN_UNITS, output_size: int = 1):
    input_layer = Input(shape=(max_seq,par_size), name="input")

    if cell_name == "LSTM":
        rnn = LSTM(layer_size)
    elif cell_name == "GRU":
        rnn = GRU(layer_size)
    elif cell_name == "NP":
        rnn = LSTM(layer_size, peephole=False)
    else:
        rnn = LSTMVariants(layer_size, cell_name)

    rnn_layer = rnn(input_layer)

    output_layer = Dense(output_size, activation="sigmoid")(rnn_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer], name=name)
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["acc"])

    model.summary()

    return model


def train_model(model, x_train, x_test, y_train, y_test, epochs: int = EPOCH, batch_size: int = BATCH_SIZE):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[RNNExperimentCallBack(model, "Dyck%d" % PAR_SIZE, batch_size, OUTPUT_DIR, EXPERIMENTS_FILE)], verbose=0)

    del model


if __name__ == "__main__":
    x, y = get_data(25000 * 6, PAR_SIZE, MIN_SEQUENCE, MAX_SEQUENCE)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, shuffle=True, test_size=.1)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    train_model(get_model("LSTM", "LSTM"), x_train, x_test, y_train, y_test)
    train_model(get_model("GRU", "GRU"), x_train, x_test, y_train, y_test)
    train_model(get_model("LSTM_NIG", "NIG"), x_train, x_test, y_train, y_test)
    train_model(get_model("LSTM_NFG", "NFG"), x_train, x_test, y_train, y_test)
    train_model(get_model("LSTM_NOG", "NOG"), x_train, x_test, y_train, y_test)
    train_model(get_model("LSTM_NP", "NP"), x_train, x_test, y_train, y_test)
    train_model(get_model("LSTM_NIAF", "NIAF"), x_train, x_test, y_train, y_test)
    train_model(get_model("LSTM_NOAF", "NOAF"), x_train, x_test, y_train, y_test)
    train_model(get_model("LSTM_FGR", "FGR"), x_train, x_test, y_train, y_test)
