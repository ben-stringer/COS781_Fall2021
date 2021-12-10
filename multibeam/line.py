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
