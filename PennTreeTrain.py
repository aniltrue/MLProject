from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from RNNLayers.LSTM import LSTM
from RNNLayers.GRU import GRU
from PennTreeReader import read_data, train_test_split

def train():
    X_data, Y_data, unique_labels, words, word2id, id2word = read_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1, shuffle=True)

    model1 = Sequential()
    model1.add(Embedding(len(words) + 1, 100, input_length=100))
    model1.add(LSTM(128))
    model1.add(Dense(len(unique_labels), activation="softmax"))

    model1.compile(optimizer=Adam(), loss=["categorical_crossentropy"], metrics=["acc"])

    model1.fit(X_data, Y_data, epochs=100, batch_size=128)

    y_pred = model1.predict(X_test)
    print("Accuracy for LSTM:", (y_pred == Y_test).mean())

    model2 = Sequential()
    model2.add(Embedding(len(words) + 1, 100, input_length=100))
    model2.add(GRU(128))
    model2.add(Dense(len(unique_labels), activation="softmax"))

    model2.compile(optimizer=Adam(), loss=["categorical_crossentropy"], metrics=["acc"])

    model2.fit(X_data, Y_data, epochs=100, batch_size=128)

    y_pred = model2.predict(X_test)
    print("Accuracy for GRU:", (y_pred == Y_test).mean())

if __name__ == "__main__":
    train()