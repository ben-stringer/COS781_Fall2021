import numpy as np
from skspatial.objects import Line, Point
import math



def main_1():
    from multibeam.arrivals import _generate_amplitude_sequence
    from multibeam.bty2d import Bty2D

    # bty = Bty2D.parse("../multibeam/bathymetry2d.bty")
    # plt.figure(1)
    # bty.plot_bathy(plt.gca())

    # src = (50, 50)
    # rec_x = np.linspace(49, 51, 10)
    # rec_y = np.full(10, 50)
    # plt.plot(rec_x, rec_y, 'r.')
    # plt.plot(src[0], src[1], 'g*')
    # plt.plot((50, 50), (50, bty.z[500]), 'b')
    # plt.plot((50, 100), (50, bty.z[-1]), 'g')

    # amp_seq = _generate_amplitude_sequence(0.001*10000, 4096)

    # plt.figure()
    # ax = plt.gca()
    # plt.xlim(0, 4608)
    # ax.set_xticks([])
    # ax.text(0, -1.2, 't0')
    # ax.text(4608, -1.2, 'tn')
    # ax.set_yticks([])
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Amplitude")
    # plt.plot(range(0, 4096), amp_seq, label="r1")
    # plt.plot(range(256, 4352), amp_seq, label="r2")
    # plt.plot(range(512, 4608), amp_seq, label="r3")
    # ax.legend()
    #
    # plt.figure()
    # ax = plt.gca()
    # plt.xlim(0, 4608)
    # ax.set_xticks([])
    # ax.text(0, -1.2, 't0')
    # ax.text(4608, -1.2, 'tn')
    # ax.set_yticks([])
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Amplitude")
    # plt.plot(range(256, 4352), amp_seq, label='r1')
    # plt.plot(range(256, 4352), amp_seq, label='r2')
    # plt.plot(range(256, 4352), amp_seq, label='r3')
    # ax.legend()

    # plt.figure()
    # ax = plt.gca()
    # plt.xlim(0, 4608)
    # ax.set_xticks([])
    # ax.text(0, -1.2, 't0')
    # ax.text(4608, -1.2, 'tn')
    # ax.set_yticks([])
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Amplitude")
    # plt.plot(range(256, 4352), amp_seq, label='r1')
    # plt.plot(range(256, 4352), amp_seq, label='r2')
    # plt.plot(range(320, 4416), amp_seq, label='r3')
    # ax.legend()
    #
    # plt.show()


def distance(line):
    return math.sqrt(
        math.pow(line.direction[0] - line.point[0], 2)
        + math.pow(line.direction[1] - line.point[1], 2))


def angle(vec):
    return math.atan(vec[1] / vec[0])


def angle_between(vec1, vec2):
    a1 = angle(vec1)
    a2 = angle(vec2)
    if a1 > a2:
        return a1-a2
    else:
        return a2-a1


def rotate(vec, th):
    x = vec[0]
    y = vec[1]
    return Point([
        math.cos(th) * x - math.sin(th) * y,
        math.sin(th) * x + math.cos(th) * y])


def main():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.invert_zaxis()

    depth = 1000

    # x, y
    plt.plot([-10, 10], [0, 0], [0, 0], color='black')
    plt.plot([0, 0], [-10, 10], [0, 0], color='black')

    # x, y at z=depth
    plt.plot([-10, 10], [0, 0], [depth, depth], color='black')
    plt.plot([0, 0], [-10, 10], [depth, depth], color='black')

    # z
    plt.plot([0, 0], [0, 0], [0, depth], color='black')

    # seafloor
    x, y = np.meshgrid(
        np.arange(-10, 11),
        np.arange(-10, 11)
    )
    ax.plot_surface(x, y, np.full((21, 21), depth), color='brown', alpha=0.7)

    # p and v
    v = rotate([0, 8], math.pi/6)
    plt.plot([0, v[0]], [0, v[1]])
    plt.plot([0, v[0]], [0, v[1]], [0, depth])
    plt.plot([v[0], v[0]], [v[1], v[1]], [0, depth], linestyle='dashed')
    ax.scatter(v[0], v[1], depth)

    plt.show()


if __name__ == "__main__":
    main()
