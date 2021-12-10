import math

import re
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

import logging

logging.basicConfig(format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
                    level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Bty3D:

    @staticmethod
    def parse(file_name):
        """
        Parse a OALib bty file
        :param file_name: the name of a file formatted in the OALib format
        :return: a Bty2D object containing the values from the file
        """
        logger.debug(f"Attempting to parse 3D bathymetry file '{file_name}'")
        with open(file_name, 'r') as fin:
            interp = fin.readline().strip('"\'\n')
            nx = int(fin.readline())
            xs = np.array(list(
                map(float, re.split(r'\s+', fin.readline().strip()))))
            ny = int(fin.readline())
            ys = np.array(list(
                map(float, re.split(r'\s+', fin.readline().strip()))))
            zs = []
            for _ in range(0, ny):
                zs.append(np.array(list(
                    map(float, re.split(r'\s+', fin.readline().strip())))))
        logger.debug(f"Parsing '{file_name}' succeeded")
        return Bty3D(interp, xs, ys, np.array(zs))

    @staticmethod
    def gen_random():
        """
        Taken from a public forum somewhere but I can't find it now.
        :return:
        """
        avg_depth = 2600
        depth_spread = 2000
        num = 1000

        F = 3

        X, Y = np.meshgrid(np.linspace(0, num, num), np.linspace(0, num, num))

        i = np.min([X - 1, num - X + 1], axis=0)
        j = np.min([Y - 1, num - Y + 1], axis=0)
        H = np.exp(-.5 * (np.square(i) + np.square(j)) / np.square(np.ones((num, num)) * F))
        Z = (np.fft.ifft2(H * np.fft.fft2(np.random.randn(num, num)))).real * depth_spread + avg_depth

        return Bty3D('L', np.linspace(0, 1000, num), np.linspace(0, 1000, num), Z)

    def __init__(self, interp, x, y, z):
        self.interp = interp
        self.x = x
        self.y = y
        self.z = z
        self.len_x = len(x)
        self.len_y = len(y)
        self.len_z = self.len_x * self.len_y

    def __str__(self):
        """
        Convert the bathymetry to a string format compatible with the OALib .bty format.
        :return: a string that can be written to a file
        """
        return "'{}'\n{}\n{}\n{}\n{}\n{}".format(
            self.interp,
            self.len_x,
            ' '.join(list(map(str, self.x))),
            self.len_y,
            ' '.join(list(map(str, self.y))),
            '\n'.join([' '.join(str(z) for z in zy) for zy in self.z])
        )

    def plot_bathy(self, fig, axis):
        X, Y = np.meshgrid(self.x, self.y)
        surf = axis.plot_surface(X, Y, self.z, linewidth=0,
                                 antialiased=False,
                                 cmap=cm.get_cmap('jet'))
        axis.invert_zaxis()
        axis.get_proj = lambda: np.dot(Axes3D.get_proj(axis),
                                       np.diag([1.25, 1.25, 1.25, 1]))
        fig.colorbar(surf, location='left', shrink=0.25)

    def to_file(self, filename):
        with open(filename, 'w') as fil:
            fil.writelines(str(self))

    def __len__(self):
        return self.len_z

    def __getitem__(self, item):
        x = int(item % self.len_x)
        y = int(item / self.len_y)
        return (self.x[x],
                self.y[y],
                self.z[x][y])

    def __iter__(self):
        return _Bty3DIterator(self)

    def gen_covering_angles(self, src_pos, num_angles_x, num_angles_y):
        """
        Create `num_angles_x * num_angles_y` (theta, phi) angle pairs.

        Theta refers to the angle around the x-y plane counterclockwise
        from the positive x-axis, ranging from 0 to 2*pi.
        Phi refers to the angle from the z-axis, ranging from 0 to pi,
        however since pi/2 refers to the x-y plane, values will never be
        larger than pi/2 and generally much smaller than this.

        The first angle pair is the angle from the source to the min-x, min-y
        (or upper-left) bathy point and the last pair is the angle from the
        source to the max-x, max-y (or lower-right) bathy point.

        :param src_pos: the point the angles start from
        :param num_angles_x: the number of equally spaced angle pairs to generate
            along the x axis
        :param num_angles_y: the number of equally spaced angle pairs to generate
            along the y axis
        :return: a numpy array of (theta, phi) angle pairs with the
            first and last angles pointing at the first and last bathymetry
            point respectively
        """
        Xi, Yi = np.meshgrid(
            np.linspace(0, self.len_x-1, num_angles_x, dtype=int),
            np.linspace(0, self.len_y-1, num_angles_y, dtype=int)
        )
        Xi = Xi.reshape(num_angles_x**2)
        Yi = Yi.reshape(num_angles_y**2)
        angles = []
        for xi, yi in zip(Xi, Yi):
            x = self.x[xi] - src_pos[0]
            y = self.y[yi] - src_pos[1]
            z = self.z[xi][yi] - src_pos[2]
            if x > 0:
                theta = math.atan(y / x)
            elif x < 0:
                theta = math.atan(y / x) + math.pi
            else:
                theta = math.pi / 2
            phi = math.atan(math.sqrt(x**2+y**2)/z)
            angles.append((theta, phi))
        return angles


class _Bty3DIterator(object):
    """
    Iterate over all bathymetry points in the supplied Bty3D.
    For the 2D case, we just created a zip of x,z values, but the
    3D case is slightly more complex, so we'll use a sub-class for it.
    """

    def __init__(self, bty):
        self.bty = bty
        self.x = 0
        self.y = 0
        # I'm precomputing the next element and updating it everytime the __next__ method is called
        self.next = (0, 0, self.bty.z[self.x][self.y])

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return the next element and determine which element should
        be returned for the subsequent call.
        :return:
        """
        if self.next is None:
            raise StopIteration
        ret = self.next
        self.x += 1
        if self.x >= self.bty.len_x:
            self.x = 0
            self.y += 1
        if self.y >= self.bty.len_y:
            self.next = None
        else:
            x = self.x
            y = self.y
            self.next = (x, y, self.bty.z[x][y])

        return ret


def main():
    bty = Bty3D.parse("bathymetry3d.bty")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    bty.plot_bathy(fig, ax)

    src = (500, 500, 50)
    num_rec_sqrt = 10
    num_rec = num_rec_sqrt**2
    rec_x, rec_y = np.meshgrid(
        np.linspace(450, 550, num_rec_sqrt),
        np.linspace(450, 550, num_rec_sqrt))
    rec_x = rec_x.reshape(num_rec)
    rec_y = rec_y.reshape(num_rec)
    rec_z = np.full(num_rec, 50)
    plt.plot(rec_x, rec_y, rec_z, 'r.')
    plt.plot(src[0], src[1], src[2], 'g*')
    plt.show()


def main_gen():
    bty = Bty3D.gen_random()
    bty.to_file("bathymetry3d.bty")


if __name__ == "__main__":
    main_gen()
    main()
