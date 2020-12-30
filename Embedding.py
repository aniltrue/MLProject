import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
import warnings
import os
import pickle as pkl

warnings.simplefilter(action='ignore', category=FutureWarning)


def getPolarity(word, data, labels):
    polarizedSize = labels.shape[1]

    results = np.zeros(polarizedSize)

    for i, sentence in enumerate(data):
        if word not in sentence:
            continue

        results[np.argmax(labels[i])] += 1

    results = results / np.sum(results)

    return results


def getEmbedding(data, labels, gloveSize: int=100, hasPolarity: bool=False, maxLength: int=64):
    embeddingIndex = {}

    glovePath = os.path.join(os.path.expanduser("~"), "glove/glove.6B.%dd.txt" % (gloveSize))

    with open(glovePath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddingIndex[word] = coefs

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    wordIndex = tokenizer.word_index

    wordIndexTemp = {}

    totalLabel = labels.shape[1]

    length = gloveSize + totalLabel if hasPolarity else gloveSize
    w2vector = np.zeros((len(wordIndex) + 1, length))

    for word, index in wordIndex.items():
        if word in embeddingIndex:
            wordIndexTemp[word] = index

            if hasPolarity:
                w2vector[index][:gloveSize] = embeddingIndex[word]
                w2vector[index][gloveSize:] = getPolarity(word, data, labels)
            else:
                w2vector[index] = embeddingIndex[word]

    wordIndex = wordIndexTemp

    layer = Embedding(w2vector.shape[0], w2vector.shape[1], input_length=maxLength, trainable=False, weights=[w2vector])

    return layer, wordIndex


def getEmbeddingFromPickle():
    wordIndexPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordIndex.pkl')
    with open(wordIndexPath, "rb") as f:
        wordIndex = pkl.load(f)

    embeddingPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embedding.pkl')
    with open(embeddingPath, "rb") as f:
        layer = pkl.load(f)

    uniqueLabelsPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uniqueLabels.pkl')
    with open(uniqueLabelsPath, "rb") as f:
        uniqueLabels = pkl.load(f)

    return layer, wordIndex, uniqueLabels