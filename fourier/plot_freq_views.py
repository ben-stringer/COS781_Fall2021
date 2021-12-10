#!/usr/bin/env python3.9

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import functions
import util


def setup_axis(ax, title, xlabel, ylabel, xlim, ylim, line):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.add_line(line)


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, t_start, t_end, N, fn, fig_num=1):
        self.t_start = t_start
        self.t_end = t_end
        self.N = N
        self.fn = fn

        self.time_ticks = np.linspace(t_start, t_end, N)
        self.signal_length = t_end - t_start
        self.freq_ticks = np.linspace(-N / 2, N / 2, N) / self.signal_length

        self.signal = fn(self.time_ticks, int(N/4))
        self.rect = np.fft.fftshift(np.fft.fft(self.signal))
        self.mag, self.phase = util.to_polar(self.rect)

        self.line_orig = Line2D(self.time_ticks, self.signal, color='black')
        self.line_complex = Line2D([], [], color='black')
        self.line_freq_real = Line2D([], [], color='black')
        self.line_freq_imag = Line2D([], [], color='black')
        self.line_mag = Line2D([], [], color='black')
        self.line_phase = Line2D([], [], color='black')

        fig = plt.figure(fig_num)
        # Only works on Linux, not Mac?
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())

        self.ax_orig = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=1)
        self.ax_complex = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
        self.ax_freq_real = plt.subplot2grid((4, 4), (0, 3))
        self.ax_freq_imag = plt.subplot2grid((4, 4), (1, 3))
        self.ax_mag = plt.subplot2grid((4, 4), (2, 3))
        self.ax_phase = plt.subplot2grid((4, 4), (3, 3))

        setup_axis(self.ax_orig, "Signal", 'Time', 'Amplitude',
                   (self.t_start, self.t_end), (fn.y_min, fn.y_max), self.line_orig)

        max_mag = max(self.mag)
        self.ax_complex.grid(True)
        setup_axis(self.ax_complex, "Fourier Transform, Complex Values", 'Real', 'Imaginary',
                   (-max_mag, max_mag), (-max_mag, max_mag), self.line_complex)

        setup_axis(self.ax_freq_real, 'Rectangular Notation Real',
                   'Frequency (Hz)', 'Amplitude', (self.freq_ticks[0], self.freq_ticks[-1]),
                   (fn.fr_min, fn.fr_max), self.line_freq_real)

        setup_axis(self.ax_freq_imag, 'Rectangular Notation Imaginary',
                   'Frequency (Hz)', 'Amplitude', (self.freq_ticks[0], self.freq_ticks[-1]),
                   (fn.fi_min, fn.fi_max), self.line_freq_imag)

        setup_axis(self.ax_mag, 'Polar Notation Magnitude', 'Frequency (Hz)',
                   'Magnitude', (self.freq_ticks[0], self.freq_ticks[-1]),
                   (0, fn.mag_max), self.line_mag)

        setup_axis(self.ax_phase, 'Polar Notation Phase', 'Frequency (Hz)',
                   'Phase (radians)', (self.freq_ticks[0], self.freq_ticks[-1]),
                   (-math.pi, math.pi), self.line_phase)

        animation.TimedAnimation.__init__(self, fig, interval=100, blit=True)

    def _draw_frame(self, framedata):
        framedata += 1
        self.line_complex.set_data(self.rect.real[0:framedata], self.rect.imag[0:framedata])
        self.line_freq_real.set_data(self.freq_ticks[0:framedata], self.rect.real[0:framedata])
        self.line_freq_imag.set_data(self.freq_ticks[0:framedata], self.rect.imag[0:framedata])
        self.line_mag.set_data(self.freq_ticks[0:framedata], self.mag[0:framedata])
        self.line_phase.set_data(self.freq_ticks[0:framedata], self.phase[0:framedata])

        self._drawn_artists = [
            self.line_complex, self.line_freq_real, self.line_freq_imag,
            self.line_mag, self.line_phase]

    def new_frame_seq(self):
        return iter(range(self.time_ticks.size - 1))

    def _init_draw(self):
        for line in [
            self.line_complex, self.line_freq_real, self.line_freq_imag,
            self.line_mag, self.line_phase
        ]:
            line.set_data([], [])


def main():
    N = int(math.pow(2, 8))
    animations = [
        # [functions.ImpulseTimeShift, None],
        [functions.NormalTimeShift, None],
        # [functions.NormalGrow, None],
        # [functions.SquareGrow, None]
    ]
    for i, entry in enumerate(animations):
        entry[1] = SubplotAnimation(0, 0.5, N, entry[0](), i)
        # ani.save('test_sub.mp4')
    plt.show()


if __name__ == "__main__":
    main()

