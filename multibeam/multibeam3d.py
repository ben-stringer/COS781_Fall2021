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
from bty3d import Bty3D

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
        rec_y = [r['pos'][1] for r in arr.receivers]
        rec_z = [r['pos'][2] for r in arr.receivers]
    else:
        logger.info(f"Pickle file {arrivals_pickle_file} does not exist; creating.")

        num_rec_sqrt = 5
        num_rec = num_rec_sqrt ** 2

        src = (500, 500, 50)
        rec_x, rec_y = np.meshgrid(
            np.linspace(450, 550, num_rec_sqrt),
            np.linspace(450, 550, num_rec_sqrt))
        rec_x = rec_x.reshape(num_rec)
        rec_y = rec_y.reshape(num_rec)
        rec_z = np.full(num_rec, 50)

        arr = Arrivals(
            Bty3D.parse("bathymetry3d.bty"),
            src,
            list(zip(rec_x, rec_y, rec_z)))

        with open(arrivals_pickle_file, 'wb') as pkl_file:
            pickle.dump(arr, pkl_file)

    return src, rec_x, rec_y, rec_z, arr


def load_or_compute_shifted_and_summed_receivers(
        arrivals, src, theta, phi, sound_speed_samples_per_meter):
    """
    Load the receiver timeshift and summation from a file if it exists,
    otherwise generate and save it.
    """
    def manual_compute():
        ts = _shift_receivers(arrivals, src, theta, phi, sound_speed_samples_per_meter)
        s = sum_receiver_timeseries(ts)
        return ts, s

    time_shift_dir = 'time_shift_3d_pkl'
    if not os.path.exists(time_shift_dir):
        logger.info(f"Making time_shift pickle directory {time_shift_dir}")
        os.mkdir(time_shift_dir)

    if os.path.isdir(time_shift_dir):
        time_shift_file = f"{time_shift_dir}/time_shift_{theta:.6f}_{phi:.6f}.pkl"
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


def _shift_receivers(arrivals, src, theta, phi, sound_speed_samples_per_meter):
    unit_vector_xy = (math.cos(theta), math.sin(theta))
    receivers = []
    for i, receiver in enumerate(arrivals.receivers):
        x, y, z = receiver['pos']
        x -= src[0]
        y -= src[1]
        z -= src[2]
        if x == 0:
            beta = 0
        else:
            beta = math.atan(y/x)
        alpha = theta - beta
        mag_rec = math.sqrt(x**2 + y**2)
        ortho_proj_onto_u = mag_rec * math.cos(alpha)
        mag_proj_point = math.sqrt(
            (unit_vector_xy[0] * ortho_proj_onto_u)**2 +
            (unit_vector_xy[1] * ortho_proj_onto_u)**2
        )
        path_difference = mag_proj_point * math.sin(phi)
        timeshift = path_difference * sound_speed_samples_per_meter
        receivers.append({
            'time': receiver['time'] - timeshift,
            'amplitude': receiver['amplitude']
        })
    return receivers


def predict_for_angle(summed_arrivals, src, theta, phi,
                      sound_speed_samples_per_meter):
    max_amplitude_index = np.argmax(summed_arrivals['amplitude'])
    max_amplitude_time_sample = summed_arrivals['time'][max_amplitude_index]
    round_trip_length_meters = (
            max_amplitude_time_sample / sound_speed_samples_per_meter)
    length_meters = round_trip_length_meters / 2
    logger.debug(f"max amplitude index: {max_amplitude_index}, "
                 f"occurred at time sample: {max_amplitude_time_sample}")
    return (src[0] + length_meters * math.sin(phi) * math.cos(theta),
            src[1] + length_meters * math.sin(phi) * math.sin(theta),
            src[2] + length_meters * math.cos(phi))


def predict_bathymetry(arr, src, angles):
    logger.info("Predicting for angles")
    num_angles = len(angles)
    pred_x = []
    pred_y = []
    pred_z = []
    for i, (theta, phi) in enumerate(angles):
        logger.debug(f"Predicting for angle ({theta:.4f}, {phi:.4f}): {i/num_angles*100:.2f}%")
        x, y, z = predict_for_angle(
            load_or_compute_shifted_and_summed_receivers(
                arr, src, theta, phi, arr.sound_speed_samples_per_meter
            )['summed'],
            src, theta, phi,
            arr.sound_speed_samples_per_meter)
        pred_x.append(x)
        pred_y.append(y)
        pred_z.append(z)
    logger.info("Predicting for angles: 100%")
    return pred_x, pred_y, pred_z


def plot_view(arr, src, rec_x, rec_y, rec_z, pred_x, pred_y, pred_z):
    """
    Plot a view including the bathymetry, the source and receivers, and
    the predicted bathymetry.
    """
    logger.info("Plotting bathymetry and source/receiver locations")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    arr.bathymetry_points.plot_bathy(fig, ax)
    ax.scatter(rec_x, rec_y, rec_z, color='red', marker='.')
    ax.scatter(src[0], src[1], src[2], color='green', marker='*')

    logger.info("Plotting predictions")
    ax.scatter(pred_x, pred_y, pred_z, color='black', marker='v')


def main():
    src, rec_x, rec_y, rec_z, arr = load_or_generate_arrivals('arrivals_pickle_3d.pkl')

    plt.figure(1)
    shifted_summed_nadir = load_or_compute_shifted_and_summed_receivers(
        arr, src, 0, 0, arr.sound_speed_samples_per_meter)
    plot_timeseries_and_sum(shifted_summed_nadir['shifted'],
                            shifted_summed_nadir['summed'])

    pred_x, pred_y, pred_z = predict_bathymetry(
        arr, src, arr.bathymetry_points.gen_covering_angles(src, 10, 10))

    plot_view(arr, src, rec_x, rec_y, rec_z, pred_x, pred_y, pred_z)

    plt.show()


if __name__ == "__main__":
    main()
