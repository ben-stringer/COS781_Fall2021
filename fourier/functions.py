import math
import sys

import numpy as np


class ImpulseTimeShift:
    def __init__(self):
        self.y_min = 0
        self.y_max = 1
        self.fr_min = -1
        self.fr_max = 1
        self.fi_min = -1
        self.fi_max = 1
        self.mag_max = 2

    def __call__(self, t, i):
        left_of_impulse = i
        right_of_impulse = len(t) - i - 1
        return np.concatenate((
            np.zeros(left_of_impulse),
            np.ones(1),
            np.zeros(right_of_impulse))
        )


class SquareGrow:
    def __init__(self):
        self.y_min = 0
        self.y_max = 1
        self.fr_min = -41
        self.fr_max = 255
        self.fi_min = -41
        self.fi_max = 255
        self.mag_max = 255

    def __call__(self, t, i):
        N = len(t)
        return np.concatenate((
            np.ones(i),
            np.zeros(N-i)))


class NormalTimeShift:
    def __init__(self):
        self.y_min = 0
        self.y_max = 1
        self.fr_min = -31
        self.fr_max = 32
        self.fi_min = -31
        self.fi_max = 32
        self.mag_max = 32

    def __call__(self, t, i):
        return np.exp(
            -np.square(t-t[i])
            / (2 * math.pow(0.005, 2)))


class NormalGrow:
    def __init__(self):
        self.y_min = 0
        self.y_max = 1
        self.fr_min = -63
        self.fr_max = 174
        self.fi_min = -63
        self.fi_max = 174
        self.mag_max = 174

    def __call__(self, t, i):
        N = len(t)
        mean = t[int(math.floor(N/2))]
        step = 0.15 / N
        stddev = step * i
        num = -np.square(t-mean)
        denom = (2 * math.pow(stddev, 2))
        return np.exp(num/(denom if denom > 0 else sys.float_info.min))
