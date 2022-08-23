import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cfg as cfg
from dataset import ReplayBuffer
from sacNet.SACNetwork import SACNet
from drone_env import DroneEnv

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

if __name__ == '__main__':
    # env = DroneEnv([-0.5, -4.0, -3.0])  # eight
    # env = DroneEnv([5, 4, 0])  # parabola
    # env = DroneEnv([5, -5, -1])  # circle
    env = DroneEnv([5, 4, 0])
    state_dim = 11
    action_dim = 4
    step_range = env.max_step_distance_1d
    action_range = np.array([step_range, step_range, step_range, step_range])

    # initialization of buffer
    replay_buffer = ReplayBuffer(cfg.REPLAY_BUFFER_SIZE)
    opt_replay_buffer = ReplayBuffer(cfg.OPT_REPLAY_BUFFER_SIZE)
    # initialization of trainer
    agent = SACNet(state_dim, action_dim, action_range, cfg.HIDDEN_DIM, replay_buffer, opt_replay_buffer,
                   cfg.SOFT_Q_LR, cfg.POLICY_LR, cfg.ALPHA_LR).to(cfg.DEVICE)

    t0 = time.time()

    # training loop
    path = '_'.join(['SAC_reward1', cfg.ENV_ID])
    model_path = os.path.join('./models', path)
    if args.train:
        if os.path.exists(os.path.join(model_path, 'sac_net.pth')):
            agent.load_weights(os.path.join(model_path, 'sac_net.pth'))
            print('load weights success!')
        frame_idx = 0
        all_episode_reward = []

        # need an extra call here to make inside functions be able to use model.forward
        state = env.reset()
        state = np.array(state)
        while len(state.shape) < 1:
            state = env.reset()
        agent.policy_net(transforms(np.reshape(state, [1, -1])))
        fig = None

        for episode in range(cfg.TRAIN_EPISODES):
            # train params init
            env.target_id = 0
            env.neg_cnt = 0
            env.steps = 1
            env.max_dist = .2
            env.min_l_pos2end = 1000
            episode_reward = 0
            temporary_buffer = []

            # init state
            env.pre_pos = np.zeros([3])
            env.pos_state_absolute = env.start
            # env.moment_state = np.zeros([12])
            env.short_steps = 0.
            env.trajectory_steps = 0.
            # TODO temporarily block train coordinates 
            env.target_object_position = np.array([1., 2., 0.])
            env.target_trajectories, env.next_target_trajectories, env.target_trajectories_all = env.get_init_target_trajectories()  # [100, 3]
            env.target_object_position = env.target_trajectories[-1, :]
            state = env.reset().astype(np.float32)

            # show positions
            xs = []
            ys = []
            zs = []

            xs_target = env.target_trajectories[:, 0].tolist()
            ys_target = env.target_trajectories[:, 1].tolist()
            zs_target = env.target_trajectories[:, 2].tolist()

            xs_predict = env.target_trajectories[:, 0].tolist()
            ys_predict = env.target_trajectories[:, 1].tolist()
            zs_predict = env.target_trajectories[:, 2].tolist()

            if cfg.RENDER:
                if fig is not None:
                    plt.close(fig)
                plt.ion()
                fig = plt.figure()

            for step in range(cfg.MAX_STEPS):
                if frame_idx > cfg.EXPLORE_STEPS:
                    action = agent.policy_net.get_action(state)
                else:
                    action = agent.policy_net.sample_action()

                next_state, reward, done, _ = env.step(None, action)

                xs_predict = env.target_trajectories[:, 0].tolist()
                ys_predict = env.target_trajectories[:, 1].tolist()
                zs_predict = env.target_trajectories[:, 2].tolist()

                if next_state is None:
                    break
                next_state = next_state.astype(np.float32)
                done = 1 if done is True else 0
                print('episode: %d, step: %d, target: %d, reward: %.2f' % (episode, step, env.target_id, reward))

                replay_buffer.push(state, action, reward, next_state, done)
                temporary_buffer.append([state, action, reward, next_state, done])

                state = next_state
                episode_reward += reward
                frame_idx += 1

                if len(replay_buffer) > cfg.BATCH_SIZE:
                    for i in range(cfg.UPDATE_ITR):
                        agent.update(cfg.BATCH_SIZE, reward_scale=cfg.REWARD_SCALE, auto_entropy=cfg.AUTO_ENTROPY,
                                     target_entropy=-1. * action_dim)

                if done:
                    break

                if cfg.RENDER:
                    pos = env.pos_state_absolute
                    target_pos = env.target_object_position
                    xs.append(pos[0])
                    ys.append(pos[1])
                    zs.append(pos[2])

                    xs_target.append(target_pos[0])
                    ys_target.append(target_pos[1])
                    zs_target.append(target_pos[2])

                    xs_predict.extend(env.predict_next_pos_10[:1, 0])
                    ys_predict.extend(env.predict_next_pos_10[:1, 1])
                    zs_predict.extend(env.predict_next_pos_10[:1, 2])

                    plt.clf()
                    img = fig.gca(projection='3d')
                    img.plot(xs, ys, zs, label='track object')
                    img.plot(xs_target, ys_target, zs_target, label='target object')
                    img.plot(xs_predict, ys_predict, zs_predict, label='track predict')
                    img.legend()
                    img.set_xlim3d(-1, 1)
                    img.set_ylim3d(-1, 1)
                    img.set_zlim3d(-1, 1)
                    img.set_xlabel('X axis')
                    img.set_ylabel('Y axis')
                    img.set_zlabel('Z axis')

                    plt.draw()
                    plt.pause(0.1)
                    plt.ioff()

            if env.target_id > env.max_target_id:
                for temp in temporary_buffer:
                    opt_replay_buffer.push(*temp)

            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, cfg.TRAIN_EPISODES, episode_reward, time.time() - t0
                )
            )

            if episode > 0 and episode % 10 == 0:
                agent.save(model_path)

        agent.save(model_path)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', path))

    # test loop
    if args.test:
        if os.path.exists(os.path.join(model_path, 'sac_net.pth')):
            agent.load_weights(os.path.join(model_path, 'sac_net.pth'))
            print('load weights success!')
        # need an extra call here to make inside functions be able to use model.forward
        state = env.reset().astype(np.float32)
        print(state)
        agent.policy_net(torch.from_numpy(state.reshape([1, -1])))
        fig = None

        for episode in range(cfg.TEST_EPISODES):
            # test param init
            env.target_id = 0
            env.neg_cnt = 0
            env.steps = 1
            env.max_dist = .2
            env.min_l_pos2end = 1000
            episode_reward = 0

            # init state
            # env.pre_pos = np.zeros([3])
            env.pos_state_absolute = env.start
            env.pre_pos = env.start
            env.moment_state = np.zeros([12])
            env.short_steps = 0.
            env.trajectory_steps = 0.
            env.target_object_position = np.array([1., 2., 0.])
            env.target_trajectories, env.next_target_trajectories, env.target_trajectories_all = env.get_init_target_trajectories()  # [100, 3]
            env.target_object_position = env.target_trajectories[-1, :]
            state = env.reset().astype(np.float32)

            # show positions
            xs = []
            ys = []
            zs = []

            xs_target = []
            ys_target = []
            zs_target = []

            # xs_target = env.target_trajectories[:, 0].tolist()
            # ys_target = env.target_trajectories[:, 1].tolist()
            # zs_target = env.target_trajectories[:, 2].tolist()

            pre_xs_target = env.target_trajectories[:, 0].tolist()
            pre_ys_target = env.target_trajectories[:, 1].tolist()
            pre_zs_target = env.target_trajectories[:, 2].tolist()

            xs_predict = env.target_trajectories[:, 0].tolist()
            ys_predict = env.target_trajectories[:, 1].tolist()
            zs_predict = env.target_trajectories[:, 2].tolist()

            if episode % 1 == 0 and cfg.RENDER:
                if fig is not None:
                    plt.close(fig)
                plt.ion()
                fig = plt.figure(figsize=(5, 4))
                # pass

            for step in range(cfg.MAX_STEPS):
                action = agent.policy_net.get_action(state, greedy=True)
                f_action = None
                # if env.steps >= 2 or env.target_id > 0:
                #     f_action = env.getFAction(action)

                state, reward, done, info = env.step(f_action, action)
                pos = env.pos_state_absolute
                print('z: ', pos[2])
                episode_reward += reward
                if done:
                    break

                print('episode: %d, step: %d, target: %d, reward: %.2f' % (episode, step, env.target_id, reward))

                if (env.target_id > 0 or env.step == 400-250) and cfg.RENDER:
                    # plt.plot(xs[-1], ys[-1], zs[-1], 'r--X', markersize=15)
                    # plt.savefig(f'./image/parabola_angle1.pdf')
                    # plt.waitforbuttonpress()
                    break

                xs_predict = env.target_trajectories[:, 0].tolist()
                ys_predict = env.target_trajectories[:, 1].tolist()
                zs_predict = env.target_trajectories[:, 2].tolist()

                if episode % 1 == 0 and cfg.RENDER:
                    pos = env.pos_state_absolute
                    target_pos = env.target_object_position
                    xs.append(pos[0])
                    ys.append(pos[1])
                    zs.append(pos[2])

                    xs_target.append(target_pos[0])
                    ys_target.append(target_pos[1])
                    zs_target.append(target_pos[2])

                    xs_predict.extend(env.predict_next_pos_10[:, 0])
                    ys_predict.extend(env.predict_next_pos_10[:, 1])
                    zs_predict.extend(env.predict_next_pos_10[:, 2])

                    # TODO plt
                    plt.clf()
                    img = fig.gca(projection='3d')
                    plt.plot(xs[0], ys[0], zs[0], 'r--p', markersize=8)
                    img.plot(xs, ys, zs, label='Interception (Agent)')
                    img.plot(xs_target, ys_target, zs_target, label='While tracked (Target)')
                    img.plot(pre_xs_target, pre_ys_target, pre_zs_target, label='Before tracked (Target)')

                    # img.plot(xs_predict, ys_predict, zs_predict, label='track predict')
                    img.legend()
                    # img.set_xlim3d(-5, 5)  # circle
                    # img.set_ylim3d(-5, 5)
                    # img.set_zlim3d(-1, 3)

                    img.set_xlim3d(-5, 5) # eight
                    img.set_ylim3d(-5, 5)
                    img.set_zlim3d(-5, 4)

                    # img.set_xlim3d(-5, 5)  # parabola
                    # img.set_ylim3d(-5, 5)
                    # img.set_zlim3d(-1, 10)
                    img.set_xlabel('X axis')
                    img.set_ylabel('Y axis')
                    img.set_zlabel('Z axis')

                    plt.draw()
                    plt.pause(0.01)
                    plt.ioff()

            pre_xs_target.append(xs_target[0])
            pre_ys_target.append(ys_target[0])
            pre_zs_target.append(zs_target[0])

            save_data = [pre_xs_target, pre_ys_target, pre_zs_target, xs_target, ys_target, zs_target, xs, ys, zs]
            np.save('./data_temp/parabola_own', save_data, allow_pickle=True, fix_imports=True)

            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, cfg.TEST_EPISODES, episode_reward,
                    time.time() - t0
                )
            )
