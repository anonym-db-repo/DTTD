import numpy as np
import matplotlib.pyplot as plt


def getCirclePts_left(radius, position, incline_angle, nums):
    pts = []
    angle = 0
    pos_x, pos_y, pos_z = position
    incline_y = np.cos(incline_angle)
    incline_z = np.sin(incline_angle)
    for _ in range(nums):
        angle += np.pi * 2 / nums
        x = radius * np.cos(angle) + pos_x
        y = radius * np.sin(angle) * incline_y + pos_y
        z = radius * np.sin(angle) * incline_z + pos_z
        pts.append(np.array([x, y, z]))
    return pts


def getCirclePts_right(radius, position, incline_angle, nums):
    pts = []
    angle = np.pi
    pos_x, pos_y, pos_z = position
    incline_y = np.cos(incline_angle)
    incline_z = np.sin(incline_angle)
    for i in range(nums):
        angle -= np.pi * 2 / nums
        x = radius * np.cos(angle) + pos_x
        y = radius * np.sin(angle) * incline_y + pos_y
        z = radius * np.sin(angle) * incline_z + pos_z
        pts.append(np.array([x, y, z]))
    return pts


def generate_data(init_radius, radius_add, init_pos, init_incline_angle, seq_len, data_len):
    datas = []
    for i in range(data_len):
        radius_new = init_radius + i * radius_add
        incline_angle = init_incline_angle + i * (np.pi / data_len)
        pts = gen_data(radius_new, init_pos, incline_angle, seq_len)
        datas.append(pts)
    return datas


def gen_data(radius, init_pos, incline_angle, seq_len):
    pts_left = getCirclePts_left(radius, init_pos, incline_angle, seq_len)
    pts_right = getCirclePts_right(radius, [init_pos[0]+2*radius, init_pos[1], init_pos[2]], incline_angle, seq_len)
    pts = []
    pts.extend(pts_left)
    pts.extend(pts_right)
    return pts


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


# Start from [0,0,0] as origin; to produce a spiral define a circle in the x,y direction and 0.01 increments in z
if __name__ == '__main__':
    # data = generate_data(0.6, 0.2, [0., 0., 0.], np.pi / 6, 400, 4000)
    # np.save('./data/eight_test_data.py', data, allow_pickle=True, fix_imports=True)

    test_data = generate_data(2, 0.2, [0., 0., 0.], np.pi / 6, 200, 2)
    # np.save('../data/eight_test_data_2', test_data, allow_pickle=True, fix_imports=True)
    data = np.load('../data/eight_test_data_2.npy')[0]
    print(data.shape)
    show_data(data)
