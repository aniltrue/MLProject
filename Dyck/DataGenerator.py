import numpy as np
import random as rnd


def get_data(size: int, par_size: int, min_size: int, max_size: int):
    RANDOM_STATE = 234
    rnd.seed(RANDOM_STATE)

    x = []
    y = []

    for i in range(size):
        _x, _y = single_data_generator(min_size, max_size, par_size)
        x.append(_x)
        y.append(_y)

    return auto_crop(x, y, max_size, par_size)


def auto_crop(x: list, y: list, max_seq: int, par_size: int):
    counter_1 = 0

    for i in range(len(y)):
        if y[i] == 1.:
            counter_1 += 1

    new_x = []
    new_y = []
    counter_0 = 0

    for i, _x in enumerate(x):
        if y[i] == 1.:
            new_x.append(_x)
            new_y.append(1.)
        elif counter_0 < counter_1:
            new_x.append(_x)
            new_y.append(0.)
            counter_0 += 1

    return generate_sequence(max_seq, new_x, par_size), np.array(new_y)


def generate_sequence(max_seq: int, x: list, par_size: int):
    result = np.zeros((len(x), max_seq, par_size))

    for i in range(len(x)):
        start_index = max_seq - len(x[i])

        for j in range(start_index, max_seq):
            result[i, j, :] = x[i][j - start_index]

    return result


def single_data_generator(min_size: int, max_size: int, par_size: int):
    result = []

    p_s = [1. / par_size]

    for i in range(1, par_size):
        p_s.append(p_s[-1] + 1. / par_size)

    for i in range(max_size):
        p1 = 0.5 if i < min_size else 0.33
        p2 = 1.0 if i < min_size else 0.66

        p = rnd.random()
        _p = rnd.random()

        if p < p1:
            par_type = np.searchsorted(p_s, _p)
            one_hot = np.identity(par_size)[par_type]
            result.append(one_hot)
        elif p < p2:
            par_type = np.searchsorted(p_s, _p)
            one_hot = -np.identity(par_size)[par_type]
            result.append(one_hot)
        else:
            break

    out = 1. if np.all(np.sum(result, axis=-2) == 0.) else 0.

    return result, out


if __name__ == "__main__":
    k = 3
    MIN_SIZE = 4

    x, y = get_data(25000 * 6, k, MIN_SIZE, 20)

    print(x.shape)
    print(y.shape)
    print(np.unique(y, return_counts=True))