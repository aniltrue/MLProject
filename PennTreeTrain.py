from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from RNNLayers.LSTM import LSTM, BiLSTM
from RNNLayers.GRU import GRU, BiGRU
from PennTreeReader import read_data, train_test_split

def train():
    X_data, Y_data, unique_labels, words, word2id, id2word = read_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1, shuffle=True)

    model1 = Sequential()
    model1.add(Embedding(len(words) + 1, 100, input_length=100))
    model1.add(LSTM(64))
    model1.add(Dense(len(unique_labels), activation="softmax"))

    model1.compile(optimizer=Adam(), loss=["categorical_crossentropy"], metrics=["acc"])

    model1.fit(X_data, Y_data, epochs=10, batch_size=128)

    y_pred = model1.predict(X_test, batch_size=128)
    print("Accuracy for LSTM:", (y_pred == Y_test).mean())

    model2 = Sequential()
    model2.add(Embedding(len(words) + 1, 100, input_length=100))
    model2.add(GRU(64))
    model2.add(Dense(len(unique_labels), activation="softmax"))

    model2.compile(optimizer=Adam(), loss=["categorical_crossentropy"], metrics=["acc"])

    model2.fit(X_data, Y_data, epochs=10, batch_size=128)

    y_pred = model2.predict(X_test, batch_size=128)
    print("Accuracy for GRU:", (y_pred == Y_test).mean())

    model3 = Sequential()
    model3.add(Embedding(len(words) + 1, 100, input_length=100))
    model3.add(BiLSTM(64))
    model3.add(Dense(len(unique_labels), activation="softmax"))

    model3.compile(optimizer=Adam(), loss=["categorical_crossentropy"], metrics=["acc"])

    model3.fit(X_data, Y_data, epochs=10, batch_size=128)

    y_pred = model3.predict(X_test, batch_size=128)
    print("Accuracy for BiLSTM:", (y_pred == Y_test).mean())

    model4 = Sequential()
    model4.add(Embedding(len(words) + 1, 100, input_length=100))
    model4.add(BiGRU(64))
    model4.add(Dense(len(unique_labels), activation="softmax"))

    model4.compile(optimizer=Adam(), loss=["categorical_crossentropy"], metrics=["acc"])

    model4.fit(X_data, Y_data, epochs=10, batch_size=128)

    y_pred = model4.predict(X_test, batch_size=128)
    print("Accuracy for BiGRU:", (y_pred == Y_test).mean())

if __name__ == "__main__":
    train()