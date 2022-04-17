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
            raise ValueError(
                f"Source and destination dimensions must be the same.  len(src) = {len_src}, len(dst) = {len_dst}")
        if len_src > 3 or len_src < 2:
            raise ValueError("Only lines of two or three dimensions supported.")
        self.src = src
        self.dst = dst
        len_fn = _length_2d if len_src == 2 else _length_3d
        self.length = len_fn(self)
        logger.debug(f"Constructed a {len_src}-D line")

    def __str__(self):
        return f'[{self.src}->{self.dst}]'


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

        def between(p, l, r):
            if l < p < r or l > p > r:
                return True
            if abs(l - p) < 10**-10 or abs(r - p) < 10**-10:
                return True
            return False

        # print(f"{x}, {y}")
        # print(f'{between(x, line1.src[0], line1.dst[0])}')
        # print(f'{between(x, line2.src[0], line2.dst[0])}')
        # print(f'{between(y, line1.src[1], line1.dst[1])}')
        # print(f'{between(y, line2.src[1], line2.dst[1])}')

        if (between(x, line1.src[0], line1.dst[0])
                and between(x, line2.src[0], line2.dst[0])
                and between(y, line1.src[1], line1.dst[1])
                and between(y, line2.src[1], line2.dst[1])):
            return x, y
        else:
            return None


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


def main():
    la = Line((50, 50), (-107.67649936828752, 2995.853487997529))
    x = [0, 20, 30, 40, 100]
    z = [1000, 1000, 800, 1000, 1000]
    bty_lines = [Line((x[i], z[i]), (x[i + 1], z[i + 1]))
                 for i in range(0, len(x) - 1)]
    for lb in bty_lines:
        print(f'{la} X {lb} : {intersection2d(la, lb)}')

    # lb = Line((40, 1000), (100, 1000))
    # lb = Line((20, 1000), (30, 800))
    # print(f'{la} X {lb} : {intersection2d(la, lb)}')


if __name__ == '__main__':
    main()
