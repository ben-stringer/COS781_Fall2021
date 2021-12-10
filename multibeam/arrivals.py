#!/usr/bin/env python3.9

import math
import matplotlib.pyplot as plt
import numpy as np
from line import Line

import logging
logging.basicConfig(format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
                    level=logging.ERROR)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


def _generate_amplitude_sequence(cycles_per_ping, samples_per_ping):
    return np.sin(
            np.linspace(
                0,
                cycles_per_ping * 2 * math.pi,
                samples_per_ping))


class Arrivals(object):

    def __init__(self, bty, src_pos, receiver_positions,
                 sound_speed_meters_per_sec=1500,
                 ping_duration_sec=0.001,
                 freq_hz=8_000):
        self.bathymetry_points = bty
        self.src_pos = src_pos
        self.receivers = [
            {
                'pos': pos,
                'arrival_times': [],
                'time': np.empty(0),
                'amplitude': np.empty(0)
            } for pos in receiver_positions]
        self.num_receivers = len(self.receivers)

        self.sound_speed_meters_per_sec = sound_speed_meters_per_sec
        self.ping_duration_sec = ping_duration_sec
        self.freq_hz = freq_hz

        self.sample_rate_hz = freq_hz * 100
        self.sound_speed_samples_per_meter = (
                self.sample_rate_hz / sound_speed_meters_per_sec)
        self.samples_per_ping = int(ping_duration_sec * self.sample_rate_hz)
        if self.samples_per_ping == 0:
            raise ValueError("Ping duration cannot be less than 10% of the frequency")
        self.cycles_per_ping = ping_duration_sec * freq_hz

        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Samples per sec: {self.sample_rate_hz}")
            logger.info(f"sound speed in samples per meter: {self.sound_speed_samples_per_meter}")
            logger.info(f"Samples per ping: {self.samples_per_ping}")
            logger.info(f"Cycles per ping: {self.cycles_per_ping}")

        self._compute_arrival_times()
        self._compute_receiver_time_series()

    def _compute_arrival_times(self):
        logger.info("Computing arrivals")
        num_pts = len(self.bathymetry_points)
        for i, bty_pt in enumerate(self.bathymetry_points):
            logger.debug(f"Computing arrivals: {i / num_pts * 100:.2f}%")
            line_src_to_btm = Line(self.src_pos, bty_pt)

            for receiver in self.receivers:
                line_btm_to_receiver = Line(bty_pt, receiver['pos'])

                round_trip_len_meters = (
                        line_src_to_btm.length +
                        line_btm_to_receiver.length)
                round_trip_time_samples_per_sec = int(
                    round_trip_len_meters *
                    self.sound_speed_samples_per_meter)

                receiver['arrival_times'].append(round_trip_time_samples_per_sec)
        logger.info("Computing arrivals: 100%")

    def _compute_receiver_time_series(self):
        logger.info("Generating receiver time series")
        for i, receiver in enumerate(self.receivers):
            logger.debug(f"Generating receiver time series: {i / self.num_receivers * 100:.2f}%")
            receiver_arrival = receiver['arrival_times']
            receiver_arrival.sort()
            arrival_i = 0
            num_arrivals = len(receiver_arrival)
            active_arrivals = []
            end = receiver_arrival[-1] + self.samples_per_ping
            x = []
            y = []
            t = receiver_arrival[0]
            while t < end or active_arrivals:

                while (arrival_i < num_arrivals and
                       receiver_arrival[arrival_i] == t):
                    active_arrivals.append(
                        iter(_generate_amplitude_sequence(
                            self.cycles_per_ping,
                            self.samples_per_ping)))
                    arrival_i += 1

                amplitude = 0
                for active in active_arrivals:
                    try:
                        amplitude += next(active)
                    except StopIteration:
                        active_arrivals.remove(active)

                x.append(t)
                y.append(amplitude)
                if active_arrivals:
                    t += 1
                elif arrival_i < num_arrivals:
                    t = receiver_arrival[arrival_i]
                else:
                    t = end

            receiver['time'] = np.array(x)
            receiver['amplitude'] = np.array(y)

        logger.info("Generating receiver time series: 100%")


def sum_receiver_timeseries(shifted_timeseries):
    all_values = []
    for i, receiver in enumerate(shifted_timeseries):
        x = receiver['time']
        y = receiver['amplitude']
        all_values.extend(zip(x, y, np.full(len(x), i, dtype=int)))
    all_values.sort()
    receiver_contribution = [0 for _ in shifted_timeseries]
    x = []
    y = []
    amplitude = 0
    for t, amp, i in all_values:
        current = receiver_contribution[i]
        amplitude = amplitude + amp - current
        receiver_contribution[i] = amp
        x.append(t)
        y.append(amplitude)

    return {
        'time': np.array(x),
        'amplitude': np.array(y)
    }


def plot_timeseries_and_sum(receivers, summed_arrivals):
    logger.info("Plotting receiver timeseries with sum")
    plt.subplot(2, 1, 1)
    plot_receiver_timeseries(receivers, alpha=0.5)
    plt.subplot(2, 1, 2)
    plot_receiver_timeseries([summed_arrivals])


def plot_receiver_timeseries(receivers, **kwargs):
    for receiver in receivers:
        plt.plot(receiver['time'], receiver['amplitude'], **kwargs)


if __name__ == "__main__":
    pass
    # main_3d()
    # main_scratch()
