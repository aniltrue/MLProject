import numpy as np
import pandas as pd
from scipy.io.wavfile import read as read_wave
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "Data/TIMIT/")

LABELS = ["g", "r", "en", "k", "epi", "d", "aa", "n", "ah", "axr", "t", "pcl", "z", "em", "iy", "ay", "ng", "ax", "ao",
          "th", "hv", "nx", "jh", "b", "s", "m", "zh", "dcl", "q", "oy", "eng", "aw", "er", "uw", "el", "ey", "uh",
          "kcl", "tcl", "dx", "pau", "l", "ih", "ax-h", "hh", "w", "ix", "f", "y", "ux", "eh", "ae", "bcl", "p", "ch",
          "dh", "ow", "gcl", "sh", "v"]


def read_csv(type: str):
    file_path = os.path.join(DATASET_DIR, "%s_data.csv" % type)

    df = pd.read_csv(file_path)
    df.dropna(inplace=True)

    audio_list = df.loc[(df["is_audio"]) & (df["is_converted_audio"]), ["path_from_data_dir"]].values.tolist()
    audio_list = [item[0] for item in audio_list]

    phonetic_list = [path[:path.index(".")] + ".PHN" for path in audio_list]

    return audio_list, phonetic_list


def one_hot_label(tag: str):
    return np.identity(len(LABELS))[LABELS.index(tag), :]


def read_file(audio_path: str, phonetic_path: str):
    max_value = np.power(2, 15) - 1  # 16bit

    with open(os.path.join(DATASET_DIR, "data/%s" % phonetic_path), "r") as f:
        phonetic_infos = f.readlines()

    sample_rate, wav = read_wave(os.path.join(DATASET_DIR, "data/%s" % audio_path))

    data = (wav.copy() + max_value) / max_value

    x = []
    y = []

    for phonetic_info in phonetic_infos:
        values = phonetic_info.split()

        start = int(values[0])
        end = int(values[1])
        tag = values[2]

        if tag not in LABELS:
            continue

        x.append(data[start:end])
        y.append(one_hot_label(tag))

    return x, y


def get_data():
    x_train, y_train = [], []
    audio_list, phonetic_list = read_csv("train")

    max_seq = 0

    for audio_path, phonetic_path in zip(audio_list, phonetic_list):
        x, y = read_file(audio_path, phonetic_path)
        x_train.extend(x)
        y_train.extend(y)

        for seq in x:
            max_seq = max(max_seq, seq.shape[0])

    print("Max. Sequence in Training:", max_seq)

    # x_train = pad_sequences(x_train, maxlen=max_seq, padding="pre", truncating="post")
    y_train = np.array(y_train, dtype="float32")

    x_test, y_test = [], []
    audio_list, phonetic_list = read_csv("test")

    max_seq_tt = 0

    for audio_path, phonetic_path in zip(audio_list, phonetic_list):
        x, y = read_file(audio_path, phonetic_path)
        x_test.extend(x)
        y_test.extend(y)

        for seq in x:
            max_seq_tt = max(max_seq_tt, seq.shape[0])

    print("Max. Sequence in Test:", max_seq_tt)

    # x_test = pad_sequences(x_test, maxlen=max_seq, padding="pre", truncating="post")
    y_test = np.array(y_test, dtype="float32")

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()

    # print(x_train.shape)
    print(y_train.shape)
    # print(x_test.shape)
    print(y_test.shape)