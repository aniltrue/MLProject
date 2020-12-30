from RNNLayers.NewGRU import NewGRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

if __name__ == "__main__":
    feature_size = 3
    seq_length = 2

    model = Sequential()
    model.add(NewGRU(32, input_shape=(seq_length, feature_size)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(), loss="mse")

    input_example = np.array([[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]]], dtype=np.float32)

    result_example = model.predict(input_example, batch_size=2)

    print(result_example)