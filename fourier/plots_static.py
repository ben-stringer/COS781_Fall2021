#!/usr/bin/env python3.9

import math
import numpy as np
import matplotlib.pyplot as plt
from util import to_polar

import logging
logging.basicConfig(format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
                    level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


test_functions = [
    ("Step function", lambda x: np.concatenate((
        np.ones(math.floor(len(x) * 0.1)),
        np.zeros(math.ceil(len(x) * 0.9))))),
    ("Impulse at x[0]", lambda x: np.concatenate((
        np.ones(1),
        np.zeros(len(x)-1)))),
    ("Impulse at x[8]", lambda x: np.concatenate((
        np.zeros(7),
        np.ones(1),
        np.zeros(len(x)-8)))),
    ("Impulse at x[N/2]", lambda x: np.concatenate((
        np.zeros(math.floor(len(x)/2)-1),
        np.ones(1),
        np.zeros(math.ceil(len(x)/2))))),
    ("sin(42\u03C9)", lambda x: np.sin(2 * math.pi * 42 * x)),
    ("sin(7\u03C9) + sin(239\u03C9)", lambda x:
            np.sin(2 * math.pi * .2 * x) +
            np.sin(2 * math.pi * .21 * x)),
    ("Gaussian, \u03C3 = 0.005", lambda x: np.exp(-np.square(x-x[int(len(x)/2)])/(2 * math.pow(0.005, 2)))),
    ("Gaussian, \u03C3 = 0.05", lambda x: np.exp(-np.square(x-x[int(len(x)/2)])/(2 * math.pow(0.05, 2)))),
]


def plot_subfig(subplot_num, x, y, title=None, xlabel=None, ylabel=None):
    """
    Plot one subfigure at the specified position
    :param subplot_num:
    :param x: the values on the x axis
    :param y: the values on the y axis
    :param xlabel: the text to use to label the x axis
    :param ylabel: the text to use to label the y axis
    """
    plt.subplot(subplot_num)
    plt.plot(x, y)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()


def plot_dft(fig_num, title, t, y, freq, f, ti, mag, phase):
    """
    Plot a FFT graph in 3 parts, the original signal, the real part of the frequency,
    and the imaginary part of the frequency.
    :param fig_num: supplied to the figure command
    :param title: the title to be placed on the figure
    :param t: array containing time intervals; the x axis for the time domain
    :param y: array containing function values for each t value
    :param freq: array containing frequency values; the x axis for the frequency domain
    :param f: array containing complex values obtained by taking the Fourier transform of y
    :param ti: array containing time intervals computed from taking the inverse
        fft of the frequency
    :param mag: array containing the magnitude in polar notation
    :param phase: array containing the phase in polar notation
    """
    plt.figure(fig_num)
    plt.suptitle(title)

    plot_subfig(321, t, y, 'Original Signal', 'Time (secs)', 'Amplitude')
    plot_subfig(323, freq, f.real, 'Rectangular Notation Real', 'Frequency (Hz)', 'Amplitude')
    plot_subfig(325, freq, f.imag, 'Rectangular Notation Imaginary', 'Frequency (Hz)', 'Amplitude')
    plot_subfig(322, t, ti, 'Inverse Fourier Signal', 'Time (secs)', 'Amplitude')
    plot_subfig(324, freq, mag, 'Polar Notation Magnitude', 'Frequency (Hz)', 'Magnitude')
    plot_subfig(326, freq, phase, 'Polar Notation Phase', 'Frequency (Hz)', 'Phase')

    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())


def plot_figures(N, t_start, t_end, idx=[k for k in range(0, len(test_functions))]):
    """
    Plot all of the test functions.
    :param N: the number of samples to take
    :param t_start: time interval start, in seconds
    :param t_end: time interval end in seconds
    :param idx: a list of the test_function indexes to plot, defaults to all
    """
    t = np.linspace(t_start, t_end, N)

    sampling_period = t_end - t_start
    sampling_rate = 1/sampling_period

    # freq = np.fft.fftshift(np.fft.fftfreq(N, sampling_period / (N - 1)))
    freq = np.linspace(-N/2, N/2, N) * sampling_rate

    for i in idx:
        title, fn = test_functions[i]
        y = fn(t)
        f = np.fft.fftshift(np.fft.fft(y))
        ti = np.fft.ifft(np.fft.ifftshift(f))
        mag, phase = to_polar(f)

        plot_dft(i, f"{title}, {sampling_period} secs, {sampling_rate} Hz {N} samples",
                 t, y, freq, f, ti, mag, phase)
    plt.show()


def main():
    logger.info("Hello world")

    t_start = 0
    t_end = 200
    N = int(math.pow(2, 10))

    plot_figures(N, t_start, t_end, [5])
    # plot_figures(N, t_start, t_end, [6, 7])


if __name__ == '__main__':
    main()
