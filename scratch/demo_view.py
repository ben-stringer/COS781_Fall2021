import numpy as np
from matplotlib import pyplot as plt

from multibeam.bty2d import Bty2D


def main():
    bty = Bty2D.parse("../multibeam/bathymetry2d.bty")
    plt.figure(1)
    bty.plot_bathy(plt.gca())

    src = (50, 50)
    rec_x = np.linspace(49, 51, 10)
    rec_y = np.full(10, 50)
    plt.plot(rec_x, rec_y, 'r.')
    plt.plot(src[0], src[1], 'g*')
    plt.plot((50, 50), (50, bty.z[500]), 'b')
    plt.plot((50, 100), (50, bty.z[-1]), 'g')

    plt.show()


if __name__ == "__main__":
    main()
