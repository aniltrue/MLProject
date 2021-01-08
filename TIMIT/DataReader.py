import numpy as np
import pandas as pd
from scipy.io.wavfile import read as read_wave
from scipy.signal import resample
import os

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


def read_file(audio_path: str, phonetic_path: str, sample_rate: int = 4000):
    max_value = np.power(2, 15) - 1  # 16bit

    with open(os.path.join(DATASET_DIR, "data/%s" % phonetic_path), "r") as f:
        phonetic_infos = f.readlines()

    _, wav = read_wave(os.path.join(DATASET_DIR, "data/%s" % audio_path))
    wav = resample(wav, sample_rate)

    data = (wav.copy() + max_value) / (2. * max_value) # Resize in [0, 1]

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


def _get_data(type: str, sample_rate: int):
    x_list, y_list = [], []
    audio_list, phonetic_list = read_csv(type)

    max_seq = 0

    for audio_path, phonetic_path in zip(audio_list, phonetic_list):
        _x, _y = read_file(audio_path, phonetic_path, sample_rate=sample_rate)
        x_list.extend(_x)
        y_list.extend(_y)

        for seq in _x:
            max_seq = max(max_seq, seq.shape[0])

    return x_list, np.array(y_list, dtype="float32"), max_seq


def generate_seq(x_list: list, seq_length: int):
    x = np.zeros((len(x_list), seq_length))

    for i, _x in enumerate(x_list):
        if _x.shape[0] > seq_length:
            x[i] = _x[_x.shape[0] - seq_length:]
        else:
            x[i, max(0, seq_length - _x.shape[0]):] = _x

    return x


def get_data(sample_rate: int = 4000):
    x_list, y_train, max_seq_tr = _get_data("train", sample_rate)
    x_train = generate_seq(x_list, max_seq_tr)

    print("Max. Sequence in Training Set:", max_seq_tr)

    x_list, y_test, max_seq_tt = _get_data("test", sample_rate)
    x_test = generate_seq(x_list, max_seq_tr)

    print("Max. Sequence in Test Set:", max_seq_tt)

        return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)