import numpy as np

import cfg
# from seq2seqNet import train, predict
from lstmCnnNet import train, predict
# from lstmNet import train, predict
from utils import MyDataset


class DroneEnv:
    def __init__(self, start):
        self.start = np.array(start) if start is not None else np.array([6., 6., -1.])
        self.pos_state_absolute = self.start  # save absolute position of the track trajectory

        # fix relationship between trajectories
        # self.z = 0
        # self.pts = getCirclePts(10)
        # self.pts = [np.array([0., 0., self.z]), np.array([4., 0., self.z]), np.array([5., 3., self.z]),
        #             np.array([2., 5., self.z]), np.array([1., 3., self.z]), np.array([0., 0., self.z])]
        # self.trajectories = self.get_trajectories()

        # train & test
        self.target_id = 0  # current track position in list of tracks
        self.steps = 1  # steps accumulated during each EPISODE
        self.neg_cnt = 0  # accumulated distance away from target position
        self.max_dist = .2  # maximum allowed distance away from trajectory track
        self.min_l_pos2end = 1000  # minimum distance between current position and target position
        self.max_target_id = 0  # max trajectory ID in each episode

        # relationship of dynamics f_action
        self.phi = 0
        # self.ddx_sign = 1
        # self.ddy_sign = 1

        # track reward
        self.velocity = 10.
        # self.dt = 0.015  # eight
        # self.dt = 0.015  # parabola
        # self.dt = 0.05  # circle
        self.dt = 0.015
        self.max_step_distance_1d = self.velocity * self.dt  # maximum allowed distance for each step
        self.max_step_distance_3d = np.sqrt(3 * self.max_step_distance_1d ** 2)  # aggregated velocity in each of 3 dims
        self.short_steps = 0.  # minimum step length for each trajectory
        self.trajectory_steps = 0.  # number of current steps for the present trajectory

        # track targets target
        # track is the controlled UAV，target is the target trajectory
        self.target_trajectories, self.next_target_trajectories, self.target_trajectories_all = self.get_init_target_trajectories()  # [100, 3]
        # train.train(self.target_trajectories_all, 'eight')
        self.predict_next_pos_10 = predict.test(self.target_trajectories[-10:, :], 'parabola')
        self.target_object_position = self.target_trajectories[-1, :]  # current target position
        self.predict_track_pos = self.get_predict_track_pos()

        # drone state
        self.pre_pos = self.start  # save previous position
        self.state = self.getState()  # state obtained from training DRL
        # self.moment_state = np.zeros([12], dtype=np.float32)  # state corresponding to f_action

    def step(self, f_action, action):
        # update target position
        if len(self.next_target_trajectories) == 0:
            return None, 0, True, None
        self.target_object_position = self.next_target_trajectories[0]
        self.target_trajectories = np.concatenate([self.target_trajectories, self.target_object_position[np.newaxis, :]],
                                               axis=0)
        self.next_target_trajectories = self.next_target_trajectories[1:, :]
        self.predict_track_pos = self.get_predict_track_pos()

        # update track position
        self.pre_pos = self.pos_state_absolute
        self.moveByPosition(action)
        # TODO f_action relationship
        # if f_action is None:
        #     self.moveByPosition(action)
        # else:
        #     self.moveByMoment(f_action)
        state_ = self.getState()
        info = None
        done = False
        reward = self.get_trajectory_reward()

        if self.isDone():
            reward = 100
            done = True
            info = 'success'

        if self.neg_cnt >= 1000:
            self.neg_cnt = 0
            done = True
            info = 'Deviate'

        self.steps += 1
        self.state = state_
        return self.state, reward, done, info

    def isDone(self):
        # determine state of trajectory
        # return True if self.target_id == len(self.trajectories) else False
        return False

    # move according to action coordinate
    def moveByPosition(self, action):
        self.pos_state_absolute = self.pos_state_absolute + action[:3]
        # self.moment_state = np.array([*self.pos_state_absolute, *action[:3], self.phi, 0, 0, 0, 0, 0])

    # get initial target trajectory
    def get_init_target_trajectories(self):
        # track_trajectories = MyDataset.drone_test_data[0, ...]  # [32, 400, 3]
        target_trajectories = MyDataset.drone_test_data[cfg.test_data_id, ...]  # [32, 400, 3]
        return target_trajectories[:250, :], target_trajectories[250:, :], target_trajectories  # [400, 3]

    # obtain target from track predict
    # def get_predict_track_pos(self):
    #     pos = self.pos_state_absolute  # current position of track
    #     pos_x, pos_y, pos_z = pos
    #     target_trajectory_last = self.target_trajectories[-10:, :]  # last 10 points of target trajectory
    #     predict_track_pos_10 = predict.test(target_trajectory_last, 'eight')  # [10, 3]
    #     self.predict_next_pos_10 = predict_track_pos_10
    #
    #     predict_track_pos_10 = np.vstack([self.target_object_position, predict_track_pos_10])
    #     # compute respective distances
    #     candidates = []
    #     # compute distance from target
    #     target_x, target_y, target_z = self.target_object_position
    #     target_dis = np.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2 + (pos_z - target_z)**2)
    #     for i in range(len(predict_track_pos_10)):
    #         next_track_x, next_track_y, next_track_z = predict_track_pos_10[i, :]
    #         euc = np.sqrt((pos_x - next_track_x)**2 + (pos_y - next_track_y)**2 + (pos_z - next_track_z)**2)
    #         # distince = euc - self.max_step_distance_3d * (i+1)
    #         if 2 * self.max_step_distance_1d > euc:
    #             candidates.append(i)
    #     track_index = min(candidates) if len(candidates) > 0 else 0
    #
    #     return predict_track_pos_10[track_index, :]  # predict next position of track

    # obtain target of track predict
    def get_predict_track_pos(self):
        pos = self.pos_state_absolute  # current position of track
        pos_x, pos_y, pos_z = pos
        target_trajectory_last = self.target_trajectories[-10:, :]  # last 100 points of target trajectory
        predict_track_pos_10 = predict.test(target_trajectory_last, 'parabola')  # [10, 3]
        self.predict_next_pos_10 = predict_track_pos_10

        # compute respective distances
        distinces = []
        # compute distance to target position
        target_x, target_y, target_z = self.target_object_position
        target_dis = np.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2 + (pos_z - target_z)**2)
        for i in range(1):
            next_track_x, next_track_y, next_track_z = predict_track_pos_10[i, :]
            euc = np.sqrt((pos_x - next_track_x)**2 + (pos_y - next_track_y)**2 + (pos_z - next_track_z)**2)
            distince = euc - self.max_step_distance_3d * (i+1)
            if distince > 0:
                distinces.append(distince)
            else:
                distinces.append(100)

        track_index = distinces.index(min(distinces))

        return np.array(predict_track_pos_10[track_index, :])
        # return np.array(predict_track_pos_10[track_index, :]) if min(distinces) < target_dis else self.target_object_position  # 预测的track的下一个位置

    def get_current_track_trajectory(self):
        # return self.pos_state_absolute, self.target_object_position
        return self.pos_state_absolute, self.get_predict_track_pos()  # start: current position of track，end: predicted coordinate

    # Derive f_action from action at different coordinates: [f_z(total lift), m1, m2, m3 (3 torque values)]
    # def getFAction(self, action):
    #     state = self.moment_state  # t
    #     pre_x, pre_y, pre_z, pre_dx, pre_dy, pre_dz = state[:6]
    #     pre_phi, pre_theta, pre_psi, pre_d_phi, pre_d_theta, pre_d_psi = state[6:]
    #
    #     dx, dy, dz = action[:3]  # t-->t+1
    #
    #     ddx = (dx - pre_dx)  # t+1
    #     ddy = (dy - pre_dy)  # t+1
    #     ddz = (dz - pre_dz)  # t+1
    #
    #     self.ddx_sign = np.sign(ddx)
    #     self.ddy_sign = np.sign(ddy)
    #
    #     cos_psi = ddx / (ddz + cfg.g)  # t+1
    #
    #     tmp = np.sqrt((ddx / cos_psi) ** 2 + ddy ** 2 / (1 - cos_psi ** 2))
    #     f_z = cfg.m * tmp  # t-->t+1
    #     cos_theta = (ddz + cfg.g) / tmp  # t+1
    #
    #     psi = np.arccos(cos_psi)  # t+1
    #     theta = np.arccos(cos_theta)  # t+1
    #
    #     d_phi = self.phi - pre_phi  # t+1
    #     d_theta = theta - pre_theta  # t+1
    #     d_psi = psi - pre_psi  # t+1
    #
    #     dd_phi = (d_phi - pre_d_phi)  # t+1
    #     dd_theta = (d_theta - pre_d_theta)  # t+1
    #     dd_psi = (d_psi - pre_d_psi)  # t+1
    #
    #     m1 = (dd_phi - pre_d_theta * pre_d_psi * ((cfg.Iy - cfg.Iz) / cfg.Ix)) * cfg.Ix  # t-->t+1
    #     m2 = (dd_theta - pre_d_phi * pre_d_psi * ((cfg.Iz - cfg.Ix) / cfg.Iy)) * cfg.Iy  # t-->t+1
    #     m3 = (dd_psi - pre_d_phi * pre_d_theta * ((cfg.Ix - cfg.Iy) / cfg.Iz)) * cfg.Iz  # t-->t+1
    #
    #     if self.steps == 2 and self.target_id == 0:
    #         pre_phi, pre_theta, pre_psi = self.phi - d_phi, theta - d_theta, psi - d_psi
    #         pre_d_phi, pre_d_theta, pre_d_psi = d_phi - dd_phi, d_theta - dd_theta, d_psi - dd_psi
    #         self.moment_state[6:] = np.array([pre_phi, pre_theta, pre_psi, pre_d_phi, pre_d_theta, pre_d_psi])
    #     return np.array([f_z, m1, m2, m3])

    # move according to f_action
    # def moveByMoment(self, f_action):
    #     state = self.moment_state  # t
    #     pre_x, pre_y, pre_z, pre_dx, pre_dy, pre_dz = state[:6]
    #     pre_phi, pre_theta, pre_psi, pre_d_phi, pre_d_theta, pre_d_psi = state[6:]
    #     f_z, m1, m2, m3 = f_action[:]  # t
    #
    #     dd_theta = m2 / cfg.Iy  # t+1
    #     dd_psi = m3 / cfg.Iz  # t+1
    #     dd_phi = 0  # t+1
    #
    #     d_phi = pre_d_phi + dd_phi  # t+1
    #     d_theta = pre_d_theta + dd_theta  # t+1
    #     d_psi = pre_d_psi + dd_psi  # t+1
    #
    #     phi = pre_phi + d_phi  # t+1
    #     theta = pre_theta + d_theta  # t+1
    #     psi = pre_psi + d_psi  # t+1
    #
    #     cos_phi, sin_phi = np.cos(phi), np.sin(phi)  # t+1
    #     cos_psi, sin_psi = np.cos(psi), np.sin(psi)  # t+1
    #     cos_theta, sin_theta = np.cos(theta), np.sin(theta)  # t+1
    #
    #     ddx = np.abs((cos_phi * cos_theta * cos_psi + sin_phi * sin_psi) * f_z / cfg.m) * self.ddx_sign  # t+1
    #     ddy = np.abs((cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) * f_z / cfg.m) * self.ddy_sign  # t+1
    #     ddz = cos_phi * cos_theta * f_z / cfg.m - cfg.g  # t+1
    #
    #     dx = pre_dx + ddx  # t+1
    #     dy = pre_dy + ddy  # t+1
    #     dz = pre_dz + ddz  # t+1
    #
    #     x = pre_x + dx  # t+1
    #     y = pre_y + dy  # t+1
    #     z = pre_z + dz  # t+1
    #
    #     state_next = np.array([x, y, z, dx, dy, dz, phi, theta, psi, d_phi, d_theta, d_psi])
    #     self.moment_state = state_next
    #     self.pos_state_absolute = state_next[:3]
    #
    #     twopi = 2 * np.pi
    #     dd_phi -= dd_phi // twopi * twopi
    #     dd_theta -= dd_theta // twopi * twopi
    #     dd_psi -= dd_psi // twopi * twopi

    # fixed trajectory: obtain list of fixed trajectory coordinates
    # def get_trajectories(self):
    #     trajectories = []
    #     for i in range(len(self.pts) - 1):
    #         trajectories.append([self.pts[i], self.pts[i + 1]])
    #     return trajectories

    # fixed trajectories: obtain fixed trajectories from target_id
    # def get_target_trajectory(self):
    #     if self.target_id >= len(self.trajectories):
    #         self.target_id = 0
    #     start, end = self.trajectories[self.target_id]
    #     return start, end

    # track new reward
    def get_trajectory_reward(self):
        self.trajectory_steps += 1
        # track 轨迹
        start, end = self.get_current_track_trajectory()
        # TODO test changing end position as current position of target
        end = self.target_object_position
        # fixed trajectory
        # start, end = self.get_target_trajectory()

        pos_state = np.array(self.pos_state_absolute)
        pos2end = end - pos_state
        start2end = end - start
        pre_pos2end = end - self.pre_pos
        l_pos2end = np.linalg.norm(pos2end)
        l_start2end = np.linalg.norm(start2end)
        l_pre_pos2end = np.linalg.norm(pre_pos2end)

        delta_dist = l_pre_pos2end - l_pos2end
        dist_line = point2line(pos_state, start, end)

        # get max_steps
        if self.short_steps < 1:
            self.short_steps = np.ceil(l_start2end / self.max_step_distance_3d)

        beta = 2.0
        beta_c = 3.0
        gamma = 2.0
        alpha = 10.0
        lamda = 2.0
        tol_rate = 10.
        closeness = np.tanh(2 * (l_start2end - l_pos2end) / l_start2end)
        # close2line = (self.max_dist - dist_line) / self.max_dist
        move_dist = np.tanh(2 * delta_dist / self.max_step_distance_1d)
        delta_steps = (self.trajectory_steps - self.short_steps) / self.short_steps - tol_rate
        step_reward = lamda * delta_steps if delta_steps > 0 else alpha * (np.exp(delta_steps) - 1)
        # step_reward = np.log(1 + abs(delta_steps / (1 + np.exp(min(-delta_steps, 16)))))
        if delta_dist < 0 or delta_steps > 100:
            # close2line = np.tanh(2 * close2line)
            reward = -np.exp(gamma + beta * (1 - closeness))
            self.neg_cnt += 1
        else:
            self.neg_cnt = 0
            move_fwd = l_pos2end < self.min_l_pos2end
            if move_fwd:
                self.min_l_pos2end = l_pos2end

            # close2line = np.tanh(2 * close2line)
            reward = np.exp(gamma + beta * closeness * move_fwd) + \
                     np.exp(gamma + beta * move_dist * move_fwd) - \
                     alpha * step_reward

                # reward = np.exp(gamma + closeness * (1 + move_dist)) - alpha * step_reward

            # else:
            #     reward = np.exp(gamma)
        print("reward: %g, closeness: %g, move_dist: %g, dist_line: %g, l_pos2end: %g, step_rwd: %g"
              % (reward, closeness, move_dist, dist_line, l_pos2end, step_reward))

        if l_pos2end < 0.5:
            self.target_id += 1
            self.min_l_pos2end = 1000
            self.trajectory_steps = 0.
            self.short_steps = 0.
            if self.target_id > self.max_target_id:
                self.max_target_id = self.target_id
            self.steps = 1

        return reward

    def reset(self):
        self.pre_pos = self.start
        self.state = self.getState()
        # TODO track add
        self.short_steps = 0.
        self.trajectory_steps = 0.
        # TODO change initial position of target coordinates
        self.target_object_position = np.array([1., 2., 0.])
        self.predict_track_pos = self.get_predict_track_pos()
        return self.state.astype(np.float32)

    def getState(self):
        pos_state = self.pos_state_absolute
        start, end = self.get_current_track_trajectory()

        start2end = end - start
        pos2end = end - pos_state
        pre_pos2end = end - self.pre_pos
        dist_line = point2line(pos_state, start, end)
        l_pos2end = np.linalg.norm(pos2end)
        l_pre_pos2end = np.linalg.norm(pre_pos2end)
        delta_l = l_pre_pos2end - l_pos2end

        state_ = np.array([*start2end, *pre_pos2end, *pos2end, dist_line, delta_l])
        return state_


# obtain distance from point to trajectory
def point2line(point, line_start, line_end):
    vec1 = line_start - point
    vec2 = line_end - point
    return np.linalg.norm(np.cross(vec1, vec2) / np.linalg.norm(line_end - line_start))


# obtain coordinates of circle from radius
def getCirclePts(radius):
    pts = []
    angle = 0
    for i in range(72):
        angle += np.pi * 2 / 72
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pts.append(np.array([x, y, 0]))
    return pts


def checkAngle(angle):
    result = False
    if -2 * np.pi <= angle <= 2 * np.pi:
        result = True
    return result
