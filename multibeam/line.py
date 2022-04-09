import math

import logging
logging.basicConfig(format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
                    level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _length_2d(line):
    return math.sqrt(
        math.pow(line.dst[0] - line.src[0], 2) +
        math.pow(line.dst[1] - line.src[1], 2))


def _length_3d(line):
    return math.sqrt(
        math.pow(line.dst[0] - line.src[0], 2) +
        math.pow(line.dst[1] - line.src[1], 2) +
        math.pow(line.dst[2] - line.src[2], 2))


class Line(object):

    def __init__(self, src, dst):
        len_src = len(src)
        len_dst = len(dst)
        if len_src != len_dst:
            raise ValueError(f"Source and destination dimensions must be the same.  len(src) = {len_src}, len(dst) = {len_dst}")
        if len_src > 3 or len_src < 2:
            raise ValueError("Only lines of two or three dimensions supported.")
        self.src = src
        self.dst = dst
        len_fn = _length_2d if len_src == 2 else _length_3d
        self.length = len_fn(self)
        logger.debug(f"Constructed a {len_src}-D line")


def intersection2d(line1, line2):
    x1 = line1.src[0]
    x2 = line1.dst[0]
    x3 = line2.src[0]
    x4 = line2.dst[0]
    y1 = line1.src[1]
    y2 = line1.dst[1]
    y3 = line2.src[1]
    y4 = line2.dst[1]

    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0.0:
        return None
    else:
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        return x, y

def angle2d(line):
    r = line.length
    x = line.dst[0] - line.src[0]
    y = line.dst[1] - line.src[1]

    if x < 0.0 and y < 0.0:
        # In quadrant 3
        return math.pi + math.asin(-y / r)
    elif y < 0.0:
        # In quadrant 4
        return 2.0 * math.pi - math.asin(-y / r)
    elif x < 0.0:
        # In quadrant 2
        return math.pi - math.asin(y / r)
    else:
        # In quadrant 1
        return math.asin(y / r)


def angle_between2d(line1, line2):
    angle1 = angle2d(line1)
    angle2 = angle2d(line2)
    if angle2 < angle2:
        angle2 += 2.0 * math.pi;
    return angle2 - angle1
