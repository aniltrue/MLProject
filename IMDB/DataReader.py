import pandas as pd
from Embedding import getEmbedding, getWordIndex, Embedding
import os
from bs4 import BeautifulSoup
import re
from nltk import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from sklearn.model_selection import train_test_split
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))), "Data/IMDB")
UNIQUE_LABEL = ["positive", "negative"]


def get_data(random_state: int = 1234, MAX_WORDS: int = 512, glove: bool = True, embedding_size: int = 100):
    nltk.download('punkt')

    path = os.path.join(DATASET_DIR, "IMDB Dataset.csv")

    df = pd.read_csv(path)

    # Denoising Text
    def denoise_text(text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = re.sub('\[[^]]*\]', '', text)

        return text

    df['review'] = df['review'].apply(denoise_text)

    texts = df['review'].to_list()
    labels = pd.get_dummies(df["sentiment"])["positive"].to_numpy()

    if glove:
        layer, word_index = getEmbedding(texts, labels, gloveSize=embedding_size, maxLength=MAX_WORDS)
    else:
        word_index = getWordIndex(texts)
        layer = Embedding(len(word_index) + 1, embedding_size, input_length=MAX_WORDS)

    sentences = []
    max_words = 0
    sum_words = 0

    for text in texts:
        words = [word.lower() for word in word_tokenize(text)]
        sentences.append(words)

        max_words = max(max_words, len(words))
        sum_words += len(words)

    print("Max. Words in a sentence:", max_words)
    print("Average Words in a sentence:", (sum_words / len(texts)))
    print("We use", MAX_WORDS)

    sequences = [[word_index.get(word, 0) for word in sentence] for sentence in sentences]

    data = pad_sequences(sequences, maxlen=MAX_WORDS, padding="pre", truncating="post")

    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=random_state)

    return X_train, X_test, Y_train, Y_test, layer, word_index


if __name__ == "__main__":  # Testing
    X_train, X_test, Y_train, Y_test, layer, word_index = get_data()

    print(len(word_index))
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    print(np.unique(Y_train, return_counts=True))
    print(np.unique(Y_test, return_counts=True))
