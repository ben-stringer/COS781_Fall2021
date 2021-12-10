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

        self.line_time = Line2D([], [], color='black')
        self.line_freq_real = Line2D([], [], color='black')
        self.line_freq_imag = Line2D([], [], color='black')
        self.line_inv = Line2D([], [], color='black')
        self.line_mag = Line2D([], [], color='black')
        self.line_phase = Line2D([], [], color='black')

        fig = plt.figure(fig_num)

        self.ax_time = fig.add_subplot(3, 2, 1)
        self.ax_freq_real = fig.add_subplot(3, 2, 3)
        self.ax_freq_imag = fig.add_subplot(3, 2, 5)
        self.ax_inv = fig.add_subplot(3, 2, 2)
        self.ax_mag = fig.add_subplot(3, 2, 4)
        self.ax_phase = fig.add_subplot(3, 2, 6)

        setup_axis(self.ax_time, "Original Signal", 'Time', 'Amplitude',
                   (t_start, t_end), (fn.y_min, fn.y_max), self.line_time)

        setup_axis(self.ax_freq_real, 'Rectangular Notation Real',
                   'Frequency (Hz)', 'Amplitude', (self.freq_ticks[0], self.freq_ticks[-1]),
                   (fn.fr_min, fn.fr_max), self.line_freq_real)

        setup_axis(self.ax_freq_imag, 'Rectangular Notation Imaginary',
                   'Frequency (Hz)', 'Amplitude', (self.freq_ticks[0], self.freq_ticks[-1]),
                   (fn.fi_min, fn.fi_max), self.line_freq_imag)

        setup_axis(self.ax_inv, 'Inverse Fourier Signal', 'Time', 'Amplitude',
                   (t_start, t_end), (fn.y_min, fn.y_max), self.line_inv)

        setup_axis(self.ax_mag, 'Polar Notation Magnitude', 'Frequency (Hz)',
                   'Magnitude', (self.freq_ticks[0], self.freq_ticks[-1]),
                   (0, fn.mag_max), self.line_mag)

        setup_axis(self.ax_phase, 'Polar Notation Phase', 'Frequency (Hz)',
                   'Phase (radians)', (self.freq_ticks[0], self.freq_ticks[-1]),
                   (-math.pi, math.pi), self.line_phase)

        animation.TimedAnimation.__init__(self, fig, interval=100, blit=True)

    def _draw_frame(self, framedata):
        y, f, ti, mag, phase = util.compute_frame_values(self.fn, self.time_ticks, framedata)

        self.line_time.set_data(self.time_ticks, y)
        self.line_freq_real.set_data(self.freq_ticks, f.real)
        self.line_freq_imag.set_data(self.freq_ticks, f.imag)
        self.line_inv.set_data(self.time_ticks, ti)
        self.line_mag.set_data(self.freq_ticks, mag)
        self.line_phase.set_data(self.freq_ticks, phase)

        self._drawn_artists = [
            self.line_time, self.line_freq_real, self.line_freq_imag,
            self.line_inv, self.line_mag, self.line_phase]

    def new_frame_seq(self):
        return iter(range(self.time_ticks.size))

    def _init_draw(self):
        for line in [
            self.line_time, self.line_freq_real, self.line_freq_imag,
            self.line_inv, self.line_mag, self.line_phase
        ]:
            line.set_data([], [])


def main():
    N = int(math.pow(2, 8))
    animations = [
        [functions.ImpulseTimeShift, None],
        [functions.NormalTimeShift, None],
        # [functions.SquareGrow, None],
        # [functions.NormalGrow, None]
    ]
    for i, entry in enumerate(animations):
        entry[1] = SubplotAnimation(0, 0.5, N, entry[0](), i)
        # ani.save('test_sub.mp4')
    plt.show()


if __name__ == "__main__":
    main()
