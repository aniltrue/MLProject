from IMDB.DataReader import get_data
from RNNLayers.LSTM import LSTM
from RNNLayers.LSTMVariants import LSTMVariants
from RNNLayers.GRU import GRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, SpatialDropout1D, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
import pickle as pkl
import os
from RNNExperimentCallBack import RNNExperimentCallBack

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/")
EXPERIMENTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments.csv")

APPLY_GLOVE = False
EMBEDDING_SIZE = 32
IS_DROP_OUT = True
DROP_OUT_RATE = 0.2
EPOCH = 25
BATCH_SIZE = 32
MAX_WORDS = 512
LAYER_SIZE = 128


def train_model(model, x_train, x_test, y_train, y_test, epochs: int = EPOCH, batch_size: int = BATCH_SIZE):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[RNNExperimentCallBack(model, "IMDB", batch_size, OUTPUT_DIR, EXPERIMENTS_FILE)])

    del model


def get_model(name: str, cell_name: str, embedding, max_words: int = MAX_WORDS, layer_size: int = LAYER_SIZE, output_size: int = 1):
    input_layer = Input(shape=(max_words,), dtype="int32", name="input")
    embedded_layer = embedding(input_layer)

    if IS_DROP_OUT:
        embedded_layer = SpatialDropout1D(DROP_OUT_RATE)(embedded_layer)

    if cell_name == "LSTM":
        rnn = LSTM(layer_size)
    elif cell_name == "GRU":
        rnn = GRU(layer_size)
    elif cell_name == "NP":
        rnn = LSTM(layer_size, peephole=False)
    else:
        rnn = LSTMVariants(layer_size, cell_name)

    rnn_layer = rnn(embedded_layer)
    if IS_DROP_OUT:
        rnn_layer = Dropout(DROP_OUT_RATE)(rnn_layer)

    output_layer = Dense(output_size, activation="sigmoid")(rnn_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer], name=name)
    model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["acc"])

    return model


def train(x_train, x_test, y_train, y_test, embedding, word_index):
    with open(os.path.join(OUTPUT_DIR, "embedding.pkl"), "wb") as f:
        pkl.dump(embedding, f)

    with open(os.path.join(OUTPUT_DIR, "word_index.pkl"), "wb") as f:
        pkl.dump(word_index, f)

    # train_model(get_model("LSTM", "LSTM", embedding), x_train, x_test, y_train, y_test)
    train_model(get_model("GRU", "GRU", embedding), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NIG", "NIG", embedding), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NFG", "NFG", embedding), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NOG", "NOG", embedding), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NP", "NP", embedding), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NIAF", "NIAF", embedding), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_NOAF", "NOAF", embedding), x_train, x_test, y_train, y_test)
    # train_model(get_model("LSTM_FGR", "FGR", embedding), x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, embedding, word_index = get_data(MAX_WORDS=MAX_WORDS, glove=APPLY_GLOVE, embedding_size=EMBEDDING_SIZE)

    train(x_train, x_test, y_train, y_test, embedding, word_index)
