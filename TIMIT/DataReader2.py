from __future__ import division, absolute_import, print_function, unicode_literals
from random import shuffle
import os

import h5py
import numpy as np
from scipy.io.wavfile import read as read_wave
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "Data/TIMIT/")


def melfb(p, n, fs):
    """
    Return a Mel filterbank matrix as a numpy array.
    Inputs:
        p:  number of filters in the filterbank
        n:  length of fft
        fs: sample rate in Hz
    Ref. www.ifp.illinois.edu/~minhdo/teaching/speaker_recognition/code/melfb.m
    """
    f0 = 700.0 / fs
    fn2 = int(np.floor(n / 2))
    lr = np.log(1 + 0.5 / f0) / (p + 1)
    CF = fs * f0 * (np.exp(np.arange(1, p + 1) * lr) - 1)
    bl = n * f0 * (np.exp(np.array([0, 1, p, p + 1]) * lr) - 1)
    b1 = int(np.floor(bl[0])) + 1
    b2 = int(np.ceil(bl[1]))
    b3 = int(np.floor(bl[2]))
    b4 = min(fn2, int(np.ceil(bl[3]))) - 1
    pf = np.log(1 + np.arange(b1, b4 + 1) / f0 / n) / lr
    fp = np.floor(pf)
    pm = pf - fp
    M = np.zeros((p, 1 + fn2))
    for c in range(b2 - 1, b4):
        r = fp[c] - 1
        M[int(r), c + 1] += 2 * (1 - pm[c])
    for c in range(b3):
        r = fp[c]
        M[int(r), c + 1] += 2 * pm[c]
    return M, CF


def dctmtx(n):
    """
    Return the DCT-II matrix of order n as a numpy array.
    """
    x, y = np.meshgrid(range(n), range(n))
    D = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * x + 1) * y / (2 * n))
    D[0] /= np.sqrt(2)
    return D


def extract(x):
    """
    Extract MFCC coefficients of the sound x in numpy array format.
    """
    FS = 16000  # Sampling rate
    FRAME_LEN = int(0.025 * FS)  # Frame length
    FRAME_SHIFT = int(0.01 * FS)  # Frame shift
    FFT_SIZE = 2048  # How many points for FFT
    WINDOW = np.hamming(FRAME_LEN)  # Window function
    PRE_EMPH = 0.97  # Pre-emphasis factor

    BANDS = 40  # Number of Mel filters
    COEFS = 13  # Number of Mel cepstra coefficients to keep
    POWER_SPECTRUM_FLOOR = 1e-100  # Flooring for the power to avoid log(0)
    M, CF = melfb(BANDS, FFT_SIZE, FS)  # The Mel filterbank matrix and the center frequencies of each band
    D = dctmtx(BANDS)[0:COEFS]  # The DCT matrix. Change the index to [0:COEFS] if you want to keep the 0-th coefficient
    invD = np.linalg.inv(dctmtx(BANDS))[:, 0:COEFS]  # The inverse DCT matrix. Change the index to [0:COEFS] if you want to keep the 0-th

    if x.ndim > 1:
        print("INFO: Input signal has more than 1 channel; the channels will be averaged.")
        x = np.mean(x, axis=1)
    frames = int((len(x) - FRAME_LEN) / FRAME_SHIFT + 1)
    feature = []
    for f in range(frames):
        # Windowing
        frame = x[f * FRAME_SHIFT: f * FRAME_SHIFT + FRAME_LEN] * WINDOW
        # Pre-emphasis
        frame[1:] -= frame[:-1] * PRE_EMPH
        # Power spectrum
        X = np.abs(np.fft.fft(frame, FFT_SIZE)[:FFT_SIZE / 2 + 1]) ** 2
        X[X < POWER_SPECTRUM_FLOOR] = POWER_SPECTRUM_FLOOR  # Avoid zero
        # Mel filtering, logarithm, DCT
        X = np.dot(D, np.log(np.dot(M, X)))
        feature.append(X)
    feature = np.row_stack(feature)

    return feature


extractor = extract
frame_size=400
frame_shift=160
derivatives=2
DTYPE = np.float64


phones = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
          'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
          'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy',
          'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau',
          'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v',
          'w', 'y', 'z', 'zh']
silence_label = phones.index('h#')

reduce_phones = {p: p for p in phones if p != 'q'}  # discard q
reduce_phones.update({
    'ae': 'aa',
    'ax': 'ah', 'ax-h': 'ah',
    'axr': 'er',
    'hv': 'hh',
    'ix': 'ih',
    'el': 'l',
    'em': 'm',
    'en': 'n', 'nx': 'n',
    'eng': 'ng',
    'zh': 'sh',
    'pcl': 'h#', 'tcl': 'h#', 'kcl': 'h#', 'bcl': 'h#', 'dcl': 'h#', 'gcl': 'h#', 'pau': 'h#', 'epi': 'h#',
    'ux': 'uw'
})



class TimitSample(object):
    @classmethod
    def create(cls, directory, name):
        f = os.path.join(directory, name.split('.')[0])
        f = f.split('/')[-4:]
        sample = cls(f[0], f[1], f[2][0], f[2][1:], f[3])
        return sample

    def __init__(self, usage, dialect, sex, speaker_id, sentence_id,
                 start=None, stop=None):
        self.usage = usage
        self.dialect = dialect
        self.sex = sex
        self.speaker_id = speaker_id
        self.sentence_id = sentence_id
        self.start = start
        self.stop = stop

    def _get_path(self, fileending):
        if not fileending.startswith('.'):
            fileending = '.' + fileending
        return os.path.join(DATASET_DIR, self.usage, self.dialect, self.sex +
                            self.speaker_id, self.sentence_id + fileending)

    def get_sentence(self):
        filename = self._get_path('txt')
        with open(filename, 'r') as f:
            content = f.read()
            start, stop, sentence = content.split(' ', 2)
            return int(start), int(stop), sentence.strip()

    def get_words(self):
        filename = self._get_path('wrd')
        with open(filename, 'r') as f:
            content = f.readlines()
            wordlist = [c.strip().split(' ', 2) for c in content]
            return [(int(start), int(stop), word)
                    for start, stop, word in wordlist
                    if (self.start is None or int(start) >= self.start) and
                       (self.stop is None or int(stop) <= self.stop)]

    def get_phones(self):
        filename = self._get_path('phn')
        with open(filename, 'r') as f:
            content = f.readlines()
            phone_list = [c.strip().split(' ', 2) for c in content]
            return [(int(start), int(stop), phone, phones.index(phone))
                    for start, stop, phone in phone_list
                    if (self.start is None or int(start) >= self.start) and
                       (self.stop is None or int(stop) <= self.stop)]

    def get_audio_data(self):
        filename = os.path.join(DATASET_DIR, self.usage, self.dialect,
                                self.sex + self.speaker_id,
                                self.sentence_id + '.wav')
        _, data = read_wave(filename)

        return data[self.start:self.stop]

    def get_labels(self, frame_size=1, frame_shift=1):
        phones = self.get_phones()
        begin = self.start if self.start else 0
        p_extended = [silence_label] * (phones[0][0] - begin)
        for p in phones:
            p_extended += [p[3]] * (int(p[1]) - int(p[0]))
        end = phones[-1][1]
        windows = zip(range(0, end - begin - frame_size + 1, frame_shift),
                      range(frame_size, end - begin + 1, frame_shift))
        labels = [np.bincount(p_extended[w[0]:w[1]]).argmax() for w in windows]
        return np.array(labels, dtype=np.byte)

    def get_features(self, extractor, frame_size=1, frame_shift=1, derivatives=0):
        d = self.get_audio_data()
        features = extractor(d)

        feature_derivs = [features]
        for i in range(derivatives):
            feature_derivs.append(np.gradient(feature_derivs[-1])[0])

        all_features = np.hstack(feature_derivs)
        labels = self.get_labels(frame_size, frame_shift)
        return all_features, labels

    def __unicode__(self):
        return '<TimitSample ' + '/'.join([self.usage, self.dialect,
                                           self.sex + self.speaker_id,
                                           self.sentence_id]) + '>'


def read_all_samples():
    samples = []
    for dirname, dirnames, filenames in os.walk(DATASET_DIR):
        samples += [TimitSample.create(dirname, n)
                    for n in filenames if n.endswith('.wav')]
    return samples


def filter_samples(samples, usage=None, dialect=None, sex=None, speaker_id=None, sentence_id=None):
    def match(s):
        return (usage is None or s.usage == usage) and \
               (dialect is None or s.dialect == dialect) and \
               (sex is None or s.sex == sex) and \
               (speaker_id is None or s.speaker_id == speaker_id) and \
                (sentence_id is None or s.sentence_id == sentence_id)
    return [s for s in samples if match(s)]