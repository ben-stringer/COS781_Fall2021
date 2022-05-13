import numpy as np
import matplotlib.pyplot as plt
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
    np.set_printoptions(linewidth=256)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.invert_zaxis()
    plt.xlabel("x")

    depth = 1000

    # x, y axis at z=0
    plt.plot([-10, 10], [0, 0], [0, 0], color='black')
    plt.plot([0, 0], [-10, 10], [0, 0], color='black')

    # x, y axis at z=depth
    plt.plot([-10, 10], [0, 0], [depth, depth], color='black')
    plt.plot([0, 0], [-10, 10], [depth, depth], color='black')

    # z
    plt.plot([0, 0], [0, 0], [0, depth], color='black')

    # seafloor
    x, y = np.meshgrid(
        np.arange(-10, 11),
        np.arange(-10, 11)
    )
    ax.plot_surface(x, y, np.full((21, 21), depth), color='brown', alpha=0.3)

    theta = math.pi/6

    # p and v
    v = rotate([0, 10], theta)
    print(f"{v}")
    vlen = math.sqrt(math.pow(v[0], 2) + math.pow(v[1], 2)) / 3
    vv = [v[0]/vlen, v[1]/vlen]
    p = (v[0], v[1], depth)
    print(f"{vlen}, {vv}")

    # origin to vv
    plt.plot([0, vv[0]], [0, vv[1]], [0, 0], color='purple')
    ax.scatter(vv[0], vv[1], 0, color='purple')
    ax.text(vv[0]-1, vv[1]-1, 0, '$v$', color='purple')
    # v
    plt.plot([-v[0], v[0]], [-v[1], v[1]], [0, 0], color='brown', linestyle='dashed')
    # origin to p
    plt.plot([0, p[0]], [0, p[1]], [0, p[2]], color='blue')
    # v to p
    plt.plot([v[0], p[0]], [v[1], p[1]], [0, p[2]], color='green', linestyle='dashed')
    # p
    ax.scatter(p[0], p[1], p[2], color="blue")
    # z to p
    plt.plot([0, p[0]], [0, p[1]], [p[2], p[2]], color='green', linestyle='dashed')
    # zx to p
    plt.plot([0, p[0]], [p[1], p[1]], [p[2], p[2]], color='black', alpha=0.5)
    # zy to p
    plt.plot([p[0], p[0]], [0, p[1]], [p[2], p[2]], color='black', alpha=0.5)
    ax.text(p[0] + 1, p[1] + 1, p[2] + 1, '$p$')

    # theta
    r = 5
    angle_start = math.pi/2
    angle_end = angle_start + theta
    angle_mid = (angle_start + angle_end) / 2
    angle_pts = np.linspace(angle_start, angle_end, 100)
    plt.plot(r * np.cos(angle_pts), r * np.sin(angle_pts), color='red')
    ax.text(
        r * 1.25 * np.cos(angle_mid),
        r * 1.25 * np.sin(angle_mid),
        0,
        r'$\theta$',
        color="red")

    # phi
    x = 400 * np.cos(np.linspace(np.pi - np.pi/4, np.pi, 10))
    y = 4 * np.sin(np.linspace(np.pi - np.pi/4, np.pi, 10))
    z = np.zeros(10)

    t90 = np.pi/2
    v = np.array([x, y, z])
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    Ry = np.array([
        [np.cos(t90), 0, np.sin(t90)],
        [0, 1, 0],
        [-np.sin(t90), 0, np.cos(t90)],
    ])
    w = np.matmul(Rz, np.matmul(Ry, v))
    plt.plot(w[0], w[1], w[2], color="red")
    ax.text(
        (w[0][0] + w[0][-1]) / 2,
        (w[1][0] + w[1][-1]) / 2,
        (w[2][0] + w[2][-1]) / 2 + 100,
        r'$\varphi$',
        color="red")

    ax.text(0, 11, 0, '$x$')
    ax.text(-11, 0, 0, '$y$')
    plt.show()


if __name__ == "__main__":
    main()
