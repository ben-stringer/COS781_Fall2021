#!/usr/bin/env python3.9

import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
                    level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Bty2D:

    @staticmethod
    def parse(file_name):
        """
        Parse a OALib bty file
        :param file_name: the name of a file formatted in the OALib format
        :return: a Bty2D object containing the values from the file
        """
        logger.debug(f"Attempting to parse 2D bathymetry file '{file_name}'")
        with open(file_name, 'r') as fin:
            interp = fin.readline().strip('"\'\n')
            n = int(fin.readline())
            xs = []
            zs = []
            for i in range(0, n):
                x, z = fin.readline().split(" ", 2)
                xs.append(float(x))
                zs.append(float(z))
        logger.debug(f"Parsing '{file_name}' succeeded")
        return Bty2D(interp, xs, zs)

    @staticmethod
    def gen_random(initial_depth=2600, num_pts=100):
        rand_nums = np.random.default_rng().standard_normal(num_pts) * 5 + 3
        x = []
        z = []
        depth = initial_depth
        for i in range(num_pts):
            delta = rand_nums[i]
            tmp = depth + delta
            depth = tmp if tmp < 3000 else depth - delta
            x.append(i)
            z.append(depth)
        return Bty2D('L', x, z)

    @staticmethod
    def gen_flat(depth=2000, num_pts=100):
        return Bty2D(
            'L',
            [i for i in range(num_pts)],
            [depth for _ in range(num_pts)])

    def __init__(self, interp, x, z):
        self.interp = interp
        self.length = len(x)
        if self.length != len(z):
            raise ValueError("The x value and depth arrays must be the same length.")
        self.x = x
        self.z = z

    def __str__(self):
        """
        Convert the bathymetry to a string format compatible with the OALib .bty format.
        :return: a string that can be written to a file
        """
        return "'{}'\n{}\n{}".format(
            self.interp,
            len(self.x),
            '\n'.join(["{} {:>12}".format(x, z) for x, z in zip(self.x, self.z)])
        )

    def plot_bathy(self, axes):
        plt.plot(self.x, self.z, '#653700')
        axes.set_ylim([0, max(self.z)+100])
        axes.invert_yaxis()

    def to_file(self, filename):
        with open(filename, 'w') as fil:
            fil.writelines(str(self))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.x[item], self.z[item]

    def __iter__(self):
        return iter(zip(self.x, self.z))

    def gen_covering_angles(self, src_pos, num_angles):
        """
        Create num_angles linearly spaced angles with the first
        being the angle from the source to the left-most bathy point
        and the last being the angle from the source to the right-most
        bathy point.
        :param src_pos: the point the angles start from
        :param num_angles: the number of equally spaced angles to generate
        :return: a numpy array of num_angles equally spaced angles with the
            first and last angles pointing at the first and last bathymetry
            point respectively
        """
        left_pt = self[0]
        right_pt = self[-1]
        left_pt_shifted = (left_pt[0] - src_pos[0], left_pt[1] - src_pos[1])
        right_pt_shifted = (right_pt[0] - src_pos[0], right_pt[1] - src_pos[1])
        return np.linspace(
            math.atan(left_pt_shifted[0] / left_pt_shifted[1]),
            math.atan(right_pt_shifted[0] / right_pt_shifted[1]),
            num_angles
        )


def main():
    bty = Bty2D.parse("bathymetry2d.bty")
    plt.figure(1)
    bty.plot_bathy(plt.gca())

    src = (50, 50)
    rec_x = np.linspace(49, 51, 10)
    rec_y = np.full(10, 50)
    plt.plot(rec_x, rec_y, 'r.')
    plt.plot(src[0], src[1], 'g*')

    plt.show()


def main_gen():
    bty = Bty2D.gen_flat()
    bty.to_file("bathymetry2d_flat.bty")


if __name__ == "__main__":
    main_gen()
    #main()
