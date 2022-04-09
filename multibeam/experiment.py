

import math
import matplotlib.pyplot as plt
import numpy as np
import arrivals
import multibeam2d
from bty2d import Bty2D, main_gen
from collections import Counter

from line import Line

import logging
logging.basicConfig(
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    level=logging.ERROR)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


def main_view_individual_returns():
    bty = Bty2D.gen_flat(1000, 1000, 0.1)
    bty.to_file("bathymetry2d.bty")
    # bty = Bty2D.gen_random(1000, 1000, 0.1)
    # bty.to_file('bathymetry2d.bty')

    src, rec_x, rec_z, arr = multibeam2d.load_or_generate_arrivals(
        "arrivals_pickle_2d.pkl",
        lambda: multibeam2d._generate_arrivals({
            'src': (50, 50),
            'rec_x': np.linspace(48, 52, 5),
            'rec_z': np.full(5, 50),
            'bty_file': "bathymetry2d.bty"
        }))

    covering_angles = arr.bathymetry_points.gen_covering_angles(src, 21)

    plt.figure(1)
    multibeam2d.plot_bty(arr, src, rec_x, rec_z)

    logger.info(f"cycles per ping {arr.cycles_per_ping}, samples per ping {arr.samples_per_ping}")
    bty = arr.bathymetry_points
    receivers = arr.receivers

    plt.figure(2)
    arrival = Counter(sorted(receivers[2]['arrival_times']))
    time = list(arrival)
    count = [arrival[c] for c in arrival]

    for i in range(len(time)):
        strength = count[i]
        start_time = time[i]
        signal = [a * strength for a in arrivals._generate_amplitude_sequence(arr.cycles_per_ping, arr.samples_per_ping)]
        plt.plot(range(start_time, start_time + arr.samples_per_ping), signal)

    plt.show()


def main_view_arrival_times():
    src, rec_x, rec_z, arr = multibeam2d.load_or_generate_arrivals(
        "arrivals_pickle_2d.pkl",
        lambda: multibeam2d._generate_arrivals({
            'src': (50, 50),
            'rec_x': np.linspace(48, 52, 5),
            'rec_z': np.full(5, 50),
            'bty_file': "bathymetry2d.bty"
        }))
    receivers = arr.receivers

    plt.figure(1)
    arrival = Counter(sorted(receivers[0]['arrival_times']))
    time = list(arrival)
    count = [arrival[c] for c in arrival]
    plt.plot(time, count)

    plt.figure(2)
    arrival = Counter(sorted(receivers[1]['arrival_times']))
    time = list(arrival)
    count = [arrival[c] for c in arrival]
    plt.plot(time, count)

    plt.figure(3)
    arrival = Counter(sorted(receivers[2]['arrival_times']))
    time = list(arrival)
    count = [arrival[c] for c in arrival]
    plt.plot(time, count)

    plt.figure(4)
    arrival = Counter(sorted(receivers[3]['arrival_times']))
    time = list(arrival)
    count = [arrival[c] for c in arrival]
    plt.plot(time, count)

    plt.figure(5)
    arrival = Counter(sorted(receivers[4]['arrival_times']))
    time = list(arrival)
    count = [arrival[c] for c in arrival]
    plt.plot(time, count)

    plt.show()


def main_compute_TOFs():
    bty = Bty2D.parse('bathymetry2d.bty')
    xx = bty.x
    zz = bty.z
    n = math.floor(len(xx) / 2)

    sample_rate_hz = 300_000_000

    def do_test(depth):
        len_nadir = Line((50, 0), (50, depth)).length
        len_left = Line((50, 0), (49.9, depth)).length

        time_to_nadir = 2 * len_nadir / 1500 * sample_rate_hz
        time_to_left = 2 * len_left / 1500 * sample_rate_hz

        print(f"    {len_nadir} : {time_to_nadir}")
        print(f"    {len_left} : {time_to_left}")
        print(f"{depth} : {abs(time_to_nadir - time_to_left) > 1}")

    for i in range(500, 5500, 500):
        do_test(i)

    # main_view_arrival_times()


if __name__ == '__main__':
    main_view_individual_returns()
    # main_view_arrival_times()
    # main_compute_TOFs()
