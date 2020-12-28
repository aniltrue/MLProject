import nltk
import numpy as np
from nltk.corpus import treebank
from itertools import chain
from sklearn.model_selection import train_test_split


def one_hot(label, unique_labels):
    return np.identity(len(unique_labels))[unique_labels.index(label, 0)]


def read_data():
    treebank_tagged_sents = list(
        chain(*[[tree.pos() for tree in treebank.parsed_sents(pf)] for pf in treebank.fileids()]))

    words_list = [[tag[0] for tag in sent] for sent in treebank_tagged_sents]
    labels = [[tag[1] for tag in sent] for sent in treebank_tagged_sents]

    words = []
    max_words = 0
    for sent in words_list:
        words.extend(sent)
        max_words = max(max_words, len(sent))

    print("Max. Words:", max_words)

    seq_length = 100

    print("Seq. Length:", seq_length)

    words = list(set(words))

    print("Number of Words:", len(words))

    unique_labels = []
    for sent in labels:
        unique_labels.extend(sent)

    unique_labels = list(set(unique_labels))

    print("Number of Unique Labels:", len(unique_labels))

    word2id = {word: i + 1 for i, word in enumerate(words)}
    id2word = {i + 1: word for i, word in enumerate(words)}

    X_data = []
    Y_data = []

    for i in range(len(treebank_tagged_sents)):
        for j in range(len(words_list[i])):
            _x = [0] * max_words

            for k in range(j + 1):
                _x[j - k] = word2id[words_list[i][k]]

            _x = _x[:seq_length]
            _x.reverse()

            X_data.append(_x)
            Y_data.append(one_hot(labels[i][j], unique_labels))

    X_data = np.array(X_data, dtype=np.int32)
    Y_data = np.array(Y_data, dtype=np.float32)

    print(X_data.shape)
    print(Y_data.shape)

    return X_data, Y_data, unique_labels, words, word2id, id2word


if __name__ == "__main__":
    # nltk.download("ptb")

    read_data()

