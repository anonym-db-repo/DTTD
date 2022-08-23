import numpy as np
import matplotlib.pyplot as plt

from utils import MyDataset
import cfg


augmented = False
N = 3
dt = 0.15
# velocity = 120
velocity = 120
velocity_dt = velocity * dt


def get_vc(r_tm, v_tm):
    vc = -r_tm.dot(v_tm)/np.linalg.norm(r_tm)
    return vc


def getZemAcc(track_pos, target_pos, dt_velocity):
    r_m = track_pos
    v_m = dt_velocity
    r_t = target_pos
    v_t = dt_velocity * 0.6
    a_t = v_t * 1

    r_tm = r_t - r_m
    v_tm = v_t - v_m

    vc = get_vc(r_tm, v_tm)
    r = np.linalg.norm(r_tm)

    t_go = r / vc

    if augmented:
        zem = r_tm + v_tm*t_go + 0.5 * a_t * t_go**2
    else:
        zem = r_tm + v_tm*t_go

    acc = N * zem / t_go ** 2

    return acc


def moveByAcc(action):
    global track_absolute_pos, target_object_position, target_trajectories, next_target_trajectories

    target_object_position = next_target_trajectories[0]
    target_trajectories = np.concatenate([target_trajectories, target_object_position[np.newaxis, :]], axis=0)
    next_target_trajectories = next_target_trajectories[1:, :]

    # track 位置更新
    return track_absolute_pos + 0.5 * action[:3] * dt**2


if __name__ == '__main__':
    fig = None

    for episode in range(cfg.TEST_EPISODES):
        # track_absolute_pos = np.array([-0.5, -4.0, -3.0])
        track_absolute_pos = np.array([5., 4.0, .0])
        target_trajectories_all = MyDataset.drone_test_data[cfg.test_data_id, ...]
        target_trajectories = target_trajectories_all[:250, :]
        next_target_trajectories = target_trajectories_all[250:, :]
        target_object_position = target_trajectories[-1, :]

        # show positions
        xs = []
        ys = []
        zs = []

        pre_xs_target = target_trajectories[:, 0].tolist()
        pre_ys_target = target_trajectories[:, 1].tolist()
        pre_zs_target = target_trajectories[:, 2].tolist()

        # xs_target = target_trajectories[:, 0].tolist()
        # ys_target = target_trajectories[:, 1].tolist()
        # zs_target = target_trajectories[:, 2].tolist()
        xs_target = []
        ys_target = []
        zs_target = []

        if episode % 1 == 0 and cfg.RENDER:
            if fig is not None:
                plt.close(fig)
            plt.ion()
            fig = plt.figure(figsize=(4.5, 4))

        for step in range(len(next_target_trajectories)):
            action = getZemAcc(track_absolute_pos, target_object_position, np.array([velocity_dt, velocity_dt, velocity_dt]))
            track_absolute_pos = moveByAcc(action)

            pos2end = target_object_position - track_absolute_pos
            l_pos2end = np.linalg.norm(pos2end)
            print(l_pos2end)
            if l_pos2end < 0.5 or step == 400 - 181:
                print('Target get! step:%d' % step)
                # plt.plot(xs[-1], ys[-1], zs[-1], 'r--X', markersize=15)
                # plt.savefig(f'./image/Azem_eight.pdf')
                # plt.waitforbuttonpress()
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
                plt.plot(xs[0], ys[0], zs[0], 'r--p', markersize=8)
                img.plot(xs, ys, zs, label='Interception (Agent)')
                img.plot(xs_target, ys_target, zs_target, label='While tracked (Target)')
                img.plot(pre_xs_target, pre_ys_target, pre_zs_target, label='Before tracked (Target)')
                img.legend()
                # img.set_xlim3d(-5, 5)
                # img.set_ylim3d(-5, 5)
                # img.set_zlim3d(-1, 3)
                img.set_xlim3d(-5, 5)  # parabola
                img.set_ylim3d(-5, 5)
                img.set_zlim3d(-5, 4)
                img.set_xlabel('X axis')
                img.set_ylabel('Y axis')
                img.set_zlabel('Z axis')

                plt.draw()
                plt.pause(0.005)
                plt.ioff()
            pre_xs_target.append(xs_target[0])
            pre_ys_target.append(ys_target[0])
            pre_zs_target.append(zs_target[0])

            save_data = [pre_xs_target, pre_ys_target, pre_zs_target, xs_target, ys_target, zs_target, xs, ys, zs]
            np.save('./data_temp/eight_zem4', save_data, allow_pickle=True, fix_imports=True)

