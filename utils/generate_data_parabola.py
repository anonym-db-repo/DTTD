import matplotlib.pyplot as plt
import numpy as np


# Given two intersection points of x-axis with path, compute path
def gen_parabola(init_pos, target_pos, num_pos, a):
    init_x, init_y = init_pos
    target_x, target_y = target_pos
    pos = []
    for i in range(num_pos+1):
        x = init_x + i * (target_x - init_x) / num_pos
        y = init_y + i * (target_y / num_pos)
        z = a * (x - init_x) * (x - target_x)
        pos.append([x, y, z])
    return pos


def gen_data(init_pos, target_pos, parabola_len, init_a, add_a, length):
    data = []
    for i in range(length):
        a = init_a - i * add_a
        pos = gen_parabola(init_pos, target_pos, parabola_len, a)
        data.append(pos)
    return data


def show_data(data):
    plt.ion()
    fig = plt.figure()
    xs, ys, zs = [], [], []
    for i in range(len(data)):
        xs.append(data[i][0])
        ys.append(data[i][1])
        zs.append(data[i][2])
        plt.clf()
        img = fig.gca(projection='3d')
        img.plot(xs, ys, zs, label='parametric curve')
        img.legend()
        img.set_xlim3d(-5, 5)
        img.set_ylim3d(-5, 5)
        img.set_zlim3d(-5, 5)
        img.set_xlabel('X axis')
        img.set_ylabel('Y axis')
        img.set_zlabel('Z axis')

        plt.draw()
        plt.pause(0.005)
        plt.ioff()

    plt.show()


if __name__ == '__main__':
    # data = gen_data([-4., -4.], [5, 4.], 399, -0.5, 0.02, 2)
    # np.save('../data/parabola_train_data_2', data, allow_pickle=True, fix_imports=True)
    # show_data(data[0])

    data = np.load('../data/parabola_test_data_2.npy')
    # show_data(data[0])
    print(data[0][-1])

    # data = gen_data([2., 1.], [12, 4.], 400, -0.1, 0.02, 200)
    # np.save('./data/parabola_test_data.npy', data, allow_pickle=True, fix_imports=True)

