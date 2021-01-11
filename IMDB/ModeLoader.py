from IMDB.Train import get_model, OUTPUT_DIR, MAX_WORDS, LAYER_SIZE, get_data, APPLY_GLOVE, EMBEDDING_SIZE
import os


def load_model(name: str, cell_name: str, embedding, weights: str, max_words: int = MAX_WORDS, layer_size: int = LAYER_SIZE, output_size: int = 1):
    print("-" * 100)
    print("Model:", name)
    model = get_model(name, cell_name, embedding, max_words, layer_size, output_size)

    weights_path = os.path.join(OUTPUT_DIR, weights)

    model.load_weights(weights_path)

    model.summary()

    return model


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, embedding, word_index = get_data(MAX_WORDS=MAX_WORDS, glove=APPLY_GLOVE,
                                                                       embedding_size=EMBEDDING_SIZE)

    load_model("LSTM", "LSTM", embedding, "LSTM_IMDB_1_128_tanh_512.h5")
    load_model("GRU", "GRU", embedding, "GRU_IMDB_2_128_tanh_512.h5")
    load_model("LSTM_NIG", "NIG", embedding, "LSTM_NIG_IMDB_8_128_tanh_512.h5")
    load_model("LSTM_NFG", "NFG", embedding, "LSTM_NFG_IMDB_9_128_tanh_512.h5")
    load_model("LSTM_NOG", "NOG", embedding, "LSTM_NOG_IMDB_10_128_tanh_512.h5")
    load_model("LSTM_NP", "NP", embedding, "LSTM_NP_IMDB_4_128_tanh_512.h5")
    load_model("LSTM_NIAF", "NIAF", embedding, "LSTM_NIAF_IMDB_6_128_tanh_512.h5")
    load_model("LSTM_NOAF", "NOAF", embedding, "LSTM_NOAF_IMDB_7_128_tanh_512.h5")
    load_model("LSTM_FGR", "FGR", embedding, "LSTM_FGR_IMDB_5_128_tanh_512.h5")

