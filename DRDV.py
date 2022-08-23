import numpy as np
import matplotlib.pyplot as plt

from utils import MyDataset
import cfg

nominal_g = -3.71
g = np.array([0., 0., 2.9])
dt = 0.1
velocity = 10
velocity_dt = velocity * dt


def get_thrust(track2target, dt_velocity):
    rg = track2target  # track-target  pos2end
    vg = dt_velocity  # 速度 velocity * dt
    gamma = 0.0
    p = [gamma + np.linalg.norm(nominal_g) ** 2 / 2, 0., -2. * np.dot(vg, vg), -12. * np.dot(vg, rg),
         -18. * np.dot(rg, rg)]

    p_roots = np.roots(p)
    for i in range(len(p_roots)):
        if np.abs(np.imag(p_roots[i])) < 0.0001:
            if p_roots[i] > 0:
                t_go = np.real(p_roots[i])
                if t_go > 0:
                    a_c = -6. * rg / t_go ** 2 - 4. * vg / t_go - g
                else:
                    a_c = np.zeros(3)
    return a_c  # acceleration


def moveByAcc(action):
    global track_absolute_pos, target_object_position, target_trajectories, next_target_trajectories

    if len(next_target_trajectories) == 0:
        return None
    target_object_position = next_target_trajectories[0]
    target_trajectories = np.concatenate([target_trajectories, target_object_position[np.newaxis, :]], axis=0)
    next_target_trajectories = next_target_trajectories[1:, :]

    # update track position
    return track_absolute_pos + 0.5 * action[:3] * dt**2


if __name__ == '__main__':
    fig = None

    for episode in range(cfg.TEST_EPISODES):
        track_absolute_pos = np.array([4.0, 4.0, 4.0])
        target_trajectories_all = MyDataset.drone_test_data[cfg.test_data_id, ...]
        target_trajectories = target_trajectories_all[:100, :]
        next_target_trajectories = target_trajectories_all[100:, :]
        target_object_position = target_trajectories[-1, :]

        # show positions
        xs = []
        ys = []
        zs = []

        xs_target = target_trajectories[:, 0].tolist()
        ys_target = target_trajectories[:, 1].tolist()
        zs_target = target_trajectories[:, 2].tolist()

        if episode % 1 == 0 and cfg.RENDER:
            if fig is not None:
                plt.close(fig)
            plt.ion()
            fig = plt.figure()

        for step in range(len(next_target_trajectories)):
            pos2end = track_absolute_pos - target_object_position
            action = get_thrust(pos2end, np.array([velocity_dt, velocity_dt, velocity_dt]))
            track_absolute_pos = moveByAcc(action)

            pos2end = target_object_position - track_absolute_pos
            l_pos2end = np.linalg.norm(pos2end)
            print(l_pos2end)
            if l_pos2end < 0.5:
                print('Target get!')
                break

            if episode % 1 == 0 and cfg.RENDER:
                target_pos = target_object_position
                xs.append(track_absolute_pos[0])
                ys.append(track_absolute_pos[1])
                zs.append(track_absolute_pos[2])
                print('x: %f, y: %f, z: %f' % (track_absolute_pos[0], track_absolute_pos[1], track_absolute_pos[2]))

                xs_target.append(target_pos[0])
                ys_target.append(target_pos[1])
                zs_target.append(target_pos[2])

                plt.clf()
                img = fig.gca(projection='3d')
                img.plot(xs, ys, zs, label='track object')
                img.plot(xs_target, ys_target, zs_target, label='target object')

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

