import math
import numpy as np

import logging
logging.basicConfig(format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
                    level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


PI_HALF = math.pi / 2
epsilon = 1e-15


def to_polar(rect):
    return np.absolute(rect), np.angle(rect)


def to_cartesian(mag, phase):
    return mag * np.exp(1j*phase)


def to_radians(angle_in_deg):
    return angle_in_deg * math.pi / 180


def to_degrees(angle_in_rad):
    return angle_in_rad * 180 / math.pi


def wrap_phase(phase):
    return ((phase + PI_HALF) % math.pi) - PI_HALF


def compute_frame_values(fn, t, i):
    y = fn(t, i)
    f = np.fft.fftshift(np.fft.fft(y))
    ti = np.fft.ifft(np.fft.ifftshift(f))
    mag, phase = to_polar(f)
    return y, f, ti, mag, phase


def _compute_bounds(fn, t):
    y_min = float("inf")
    y_max = -float("inf")
    fr_min = float("inf")
    fr_max = -float("inf")
    fi_min = float("inf")
    fi_max = -float("inf")
    mag_max = -float("inf")
    for i in range(0, len(t)):
        y, f, ti, mag, phase = compute_frame_values(fn, t, i)
        y_min = min(y_min, np.min(y))
        y_max = max(y_max, np.max(y))
        fr_min = min(fr_min, np.min(f.real))
        fr_max = max(fr_max, np.max(f.real))
        fi_min = min(fr_min, np.min(f.imag))
        fi_max = max(fr_max, np.max(f.imag))
        mag_max = max(mag_max, np.max(mag))
    logger.info(f"self.y_min = {math.floor(y_min)}")
    logger.info(f"self.y_max = {math.ceil(y_max)}")
    logger.info(f"self.fr_min = {math.floor(fr_min)}")
    logger.info(f"self.fr_max = {math.ceil(fr_max)}")
    logger.info(f"self.fi_min = {math.floor(fi_min)}")
    logger.info(f"self.fi_max = {math.ceil(fi_max)}")
    logger.info(f"self.mag_max = {math.ceil(mag_max)}")
