#!/usr/bin/env python3.9

import os.path
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
from arrivals import (
    Arrivals,
    plot_timeseries_and_sum,
    sum_receiver_timeseries)
from bty2d import Bty2D

import logging
logging.basicConfig(format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
                    level=logging.ERROR)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


def load_or_generate_arrivals(arrivals_pickle_file):
    if os.path.exists(arrivals_pickle_file):
        logger.info(f"Loading arrival from pickle file {arrivals_pickle_file}.")
        with open(arrivals_pickle_file, 'rb') as pkl_file:
            arr = pickle.load(pkl_file)
        src = arr.src_pos
        rec_x = [r['pos'][0] for r in arr.receivers]
        rec_z = [r['pos'][1] for r in arr.receivers]
    else:
        logger.info(f"Pickle file {arrivals_pickle_file} does not exist; creating.")
        num_rec = 21
        src = (50, 50)
        rec_x = np.linspace(35, 65, num_rec)
        rec_z = np.full(num_rec, 50)
        arr = Arrivals(
            Bty2D.parse("bathymetry2d.bty"),
            src,
            list(zip(rec_x, rec_z)))

        with open(arrivals_pickle_file, 'wb') as pkl_file:
            pickle.dump(arr, pkl_file)

    return src, rec_x, rec_z, arr


def load_or_compute_shifted_and_summed_receivers(
        arrivals, src, theta, sound_speed_samples_per_meter):
    """
    Load the receiver timeshift and summation from a file if it exists,
    otherwise generate and save it.
    """
    def manual_compute():
        ts = _shift_receivers(arrivals, src, theta, sound_speed_samples_per_meter)
        s = sum_receiver_timeseries(ts)
        return ts, s

    time_shift_dir = 'time_shift_2d_pkl'
    if not os.path.exists(time_shift_dir):
        logger.info(f"Making time_shift pickle directory {time_shift_dir}")
        os.mkdir(time_shift_dir)

    if os.path.isdir(time_shift_dir):
        time_shift_file = f"{time_shift_dir}/time_shift_{theta:.6f}.pkl"
        if os.path.exists(time_shift_file):
            logger.debug(f"Time-shift pickle file {time_shift_file} exists.  "
                         f"Unpickling.")
            with open(time_shift_file, 'rb') as tsf:
                shifted, summed = pickle.load(tsf)
        else:
            logger.debug(f"Time-shift pickle file {time_shift_file} does not exist.  "
                         f"Computing time-shift and pickling.")
            shifted, summed = manual_compute()
            with open(time_shift_file, 'wb') as tsf:
                pickle.dump((shifted, summed), tsf)
    else:
        logger.warning(f"{time_shift_dir} is not a directory and could not be created.  "
                       f"Computing time-shift but not pickling.")
        shifted, summed = manual_compute()
    return {
        'shifted': shifted,
        'summed': summed
    }


def _shift_receivers(arrivals, src, theta, sound_speed_samples_per_meter):
    """
    Shift the receiver timeseries according to the formula for multibeam:
    delta_t = (d * sin(theta))/c
    where theta is the angle from the z-axis, d is the distance from the source,
    and c is the speed of sound.
    :param arrivals: the arrivals object which contains the receiver timeseries
    :param src: the location of the source; the origin for the angle theta
    :param theta: the angle from z to shift for
    :return: the receivers timeshifted to theta
    """
    receivers = []
    for i, receiver in enumerate(arrivals.receivers):
        distance_from_src_meters = (src[0] - receiver['pos'][0])
        path_difference = (
                distance_from_src_meters * math.sin(theta))
        timeshift = path_difference * sound_speed_samples_per_meter
        receivers.append({
            'time': receiver['time'] - timeshift,
            'amplitude': receiver['amplitude']
        })
    return receivers


def predict_for_angle(summed_arrivals, src, theta, sound_speed_samples_per_meter):
    """
    Given a timeshifted and summed receiver timeseries, locate the maximum
    point, determine the distance to it, and convert to cartesian coordinates.
    :param summed_arrivals: the receiver timeseries, summed
    :param src: the location of the source
    :param theta: the angle the arrivals were shifted for
    :param sound_speed_samples_per_meter: the sound speed in time samples per meter
    :return: the (x, z) cartesian coordinate
    """
    max_amplitude_index = np.argmax(summed_arrivals['amplitude'])
    max_amplitude_time_sample = summed_arrivals['time'][max_amplitude_index]
    round_trip_length_meters = (
            max_amplitude_time_sample / sound_speed_samples_per_meter)
    length_meters = round_trip_length_meters / 2
    logger.debug(f"max amplitude index: {max_amplitude_index}, occurred at time sample: {max_amplitude_time_sample}")
    return ((src[0] + length_meters * math.sin(theta)),
            src[1] + length_meters * math.cos(theta))


def predict_bathymetry(arr, src, angles):
    """
    Predict the bathymetry for each angle in `angles` by timeshifting
    the arrivals of each receiver, summing, and finding the maximum amplitude.
    :param arr: the Arrival object which contains the receiver timeseries
    :param src: the location of the source
    :param angles: the angles to predict for
    :return: a pair of lists representing (x, z) pairs
    """
    logger.info("Predicting for angles")
    num_angles = len(angles)
    pred_x = []
    pred_z = []
    for i, theta in enumerate(angles):
        logger.debug(f"Predicting for angle {theta}: {i/num_angles*100:.2f}%")
        x, z = predict_for_angle(
            load_or_compute_shifted_and_summed_receivers(
                arr, src, theta, arr.sound_speed_samples_per_meter
            )['summed'],
            src, theta,
            arr.sound_speed_samples_per_meter)
        pred_x.append(x)
        pred_z.append(z)
    logger.info("Predicting for angles: 100%")
    return pred_x, pred_z


def plot_view(arr, src, rec_x, rec_z, pred_x, pred_z):
    """
    Plot a view including the bathymetry, the source and receivers, and
    the predicted bathymetry.
    """
    logger.info("Plotting bathymetry and source/receiver locations")
    arr.bathymetry_points.plot_bathy(plt.gca())
    plt.plot(rec_x, rec_z, 'r.')
    plt.plot(src[0], src[1], 'g*')

    logger.info("Plotting predictions")
    plt.scatter(pred_x, pred_z)
    plt.plot(pred_x, pred_z)


def main():
    src, rec_x, rec_z, arr = load_or_generate_arrivals('arrivals_pickle_2d.pkl')

    plt.figure(1)
    shifted_summed_nadir = load_or_compute_shifted_and_summed_receivers(
        arr, src, 0, arr.sound_speed_samples_per_meter)
    plot_timeseries_and_sum(shifted_summed_nadir['shifted'],
                            shifted_summed_nadir['summed'])

    pred_x, pred_z = predict_bathymetry(
        arr, src, arr.bathymetry_points.gen_covering_angles(src, 51))

    plt.figure(2)
    plot_view(arr, src, rec_x, rec_z, pred_x, pred_z)

    plt.show()


if __name__ == "__main__":
    main()
