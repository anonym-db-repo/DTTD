import torch
import numpy as np
import os
from torch.autograd import Variable

from .SoftQNetwork import SoftQNet
from .PolicyNetwork import PolicyNet
import cfg


class SACNet(torch.nn.Module):
    def __init__(
            self, state_dim, action_dim, action_range, hidden_dim, replay_buffer, opt_replay_buffer, soft_q_lr=3e-4,
            policy_lr=3e-4, alpha_lr=3e-4, action_bias=None
    ):
        super(SACNet, self).__init__()
        self.replay_buffer = replay_buffer
        self.opt_replay_buffer = opt_replay_buffer

        # initialize all networks
        self.soft_q_net1 = SoftQNet(state_dim, action_dim, hidden_dim).to(cfg.DEVICE)
        self.soft_q_net2 = SoftQNet(state_dim, action_dim, hidden_dim).to(cfg.DEVICE)
        self.target_soft_q_net1 = SoftQNet(state_dim, action_dim, hidden_dim).to(cfg.DEVICE)
        self.target_soft_q_net2 = SoftQNet(state_dim, action_dim, hidden_dim).to(cfg.DEVICE)
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim, action_range, action_bias=action_bias).to(cfg.DEVICE)
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.train()

        self.log_alpha = torch.zeros([])
        self.log_alpha = Variable(torch.zeros([]), requires_grad=True)
        self.alpha = Variable(torch.exp(self.log_alpha))
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)
        # set mode
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.train()

        # initialize weights of target networks
        self.target_soft_q_net1 = self.target_ini(self.soft_q_net1, self.target_soft_q_net1)
        self.target_soft_q_net2 = self.target_ini(self.soft_q_net2, self.target_soft_q_net2)

        self.soft_q_optimizer1 = torch.optim.Adam(self.soft_q_net1.parameters(), soft_q_lr)
        self.soft_q_optimizer2 = torch.optim.Adam(self.soft_q_net2.parameters(), soft_q_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), policy_lr)

        self.log_alpha.requires_grad = True
        print(self.log_alpha)
        # self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], alpha_lr)

        self.soft_q_loss1 = torch.nn.MSELoss()
        self.soft_q_loss2 = torch.nn.MSELoss()

    def target_ini(self, net, target_net):
        """ hard-copy update for initializing target networks """
        for name in target_net.state_dict():
            target_net.state_dict()[name].copy_(net.state_dict()[name])
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        """ soft update the target net with Polyak averaging """
        for name in target_net.state_dict():
            target_net.state_dict()[name].copy_(target_net.state_dict()[name] * (1.0 - soft_tau) + \
                                            net.state_dict()[name] * soft_tau)
        return target_net

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        """ update all networks in SAC_2 """
        opt_size = int(np.ceil(0.5 * batch_size))
        if len(self.opt_replay_buffer) >= opt_size:
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size - opt_size)
            opt_state, opt_action, opt_reward, opt_next_state, opt_done = self.opt_replay_buffer.sample(opt_size)

            state = torch.from_numpy(np.concatenate([state, opt_state], 0))
            next_state = torch.from_numpy(np.concatenate([next_state, opt_next_state], 0))
            action = torch.from_numpy(np.concatenate([action, opt_action], 0))
            reward = np.concatenate([reward, opt_reward], 0)[:, np.newaxis]  # expand dim
            done = np.concatenate([done, opt_done], 0)[:, np.newaxis]
        else:
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

            state = torch.from_numpy(state)
            next_state = torch.from_numpy(next_state)
            action = torch.from_numpy(action)
            reward = reward[:, np.newaxis]  # expand dim
            done = done[:, np.newaxis]

        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        target_q_input = torch.cat([next_state, new_next_action], 1).to(torch.float32)  # the dim 0 is number of samples
        target_q_min = torch.min(
            self.target_soft_q_net1(target_q_input), self.target_soft_q_net2(target_q_input)
        ) - self.alpha * next_log_prob
        target_q_value = torch.tensor(reward + (1 - done) * gamma * target_q_min.detach().numpy(), dtype=torch.float32)  # if done==1, only reward
        q_input = torch.cat([state, action], 1).to(torch.float32)  # the dim 0 is number of samples

        predicted_q_value1 = self.soft_q_net1(q_input)
        q_value_loss1 = self.soft_q_loss1(predicted_q_value1, target_q_value)
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        # writer.add_scalar('scalar/q_value_loss1', q_value_loss1)

        predicted_q_value2 = self.soft_q_net2(q_input)
        q_value_loss2 = self.soft_q_loss2(predicted_q_value2, target_q_value)
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_q_input = torch.cat([state, new_action], 1).to(torch.float32)  # the dim 0 is number of samples
        """ implementation 1 """
        predicted_new_q_value = torch.min(self.soft_q_net1(new_q_input), self.soft_q_net2(new_q_input))
        policy_loss = torch.mean(self.alpha * log_prob - predicted_new_q_value)
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        
        if auto_entropy is True:
            alpha_loss = - torch.mean(self.log_alpha * (log_prob + torch.tensor(target_entropy)))
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            # alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = torch.exp(self.log_alpha)
        else:  # fixed alpha
            self.alpha = 1.
            alpha_loss = 0

        # Soft update the target value nets
        self.target_soft_q_net1 = self.target_soft_update(self.soft_q_net1, self.target_soft_q_net1, soft_tau)
        self.target_soft_q_net2 = self.target_soft_update(self.soft_q_net2, self.target_soft_q_net2, soft_tau)

    def save(self, save_path):  # save trained weights
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save({
            'model_q_net1_state_dict': self.soft_q_net1.state_dict(),
            'model_q_net2_state_dict': self.soft_q_net2.state_dict(),
            'model_target_q_net1_state_dict': self.target_soft_q_net1.state_dict(),
            'model_target_q_net2_state_dict': self.target_soft_q_net2.state_dict(),
            'model_policy_net_state_dict': self.policy_net.state_dict(),
            'log_alpha_state_dict': self.log_alpha
        }, os.path.join(save_path, 'sac_net.pth'))
        print(self.log_alpha)

    def load_weights(self, save_path):  # load trained weights
        checkpoint = torch.load(save_path)
        self.soft_q_net1.load_state_dict(checkpoint['model_q_net1_state_dict'])
        self.soft_q_net2.load_state_dict(checkpoint['model_q_net2_state_dict'])
        self.target_soft_q_net1.load_state_dict(checkpoint['model_target_q_net1_state_dict'])
        self.target_soft_q_net2.load_state_dict(checkpoint['model_target_q_net2_state_dict'])
        self.policy_net.load_state_dict(checkpoint['model_policy_net_state_dict'])
        self.log_alpha = checkpoint['log_alpha_state_dict']
        print(self.log_alpha)

#
# if __name__ == '__main__':
#     from dataset import ReplayBuffer
#     replay_buffers = ReplayBuffer(5e5)
#     sac_net = SACNet(24, 4, 1, 32, replay_buffers)
#     sac_net.update(300)
