import torch
import numpy as np
from torch.distributions import Normal


class PolicyNet(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., log_std_min=-20, log_std_max=2,
                 action_bias=None):
        super(PolicyNet, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_range = torch.from_numpy(action_range)
        self.action_bias = torch.zeros_like(self.action_range) if action_bias is None\
            else torch.from_numpy(action_bias)
        self.num_actions = num_actions

        negative_slope = 0.1
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_inputs, out_features=hidden_dim),
            torch.nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.linear3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.mean_linear = torch.nn.Linear(in_features=hidden_dim, out_features=num_actions)
        self.log_std_linear = torch.nn.Linear(in_features=hidden_dim, out_features=num_actions)

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        """ generate action with state for calculating gradients """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)

        action_0 = torch.tanh(mean + std * z)  # tanh distribution as actions;
        action = self.action_range * action_0 + self.action_bias
        # according to original paper, with an extra last term for normalizing different action range
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1. - action_0**2 + epsilon) - \
            torch.log(self.action_range)
        log_prob = torch.sum(log_prob, 1).reshape([-1, 1])  # expand dim as reduce_sum causes 1 dim reduced
        return action, log_prob, z, mean, log_std

    def get_action(self, state, greedy=False):
        """ generate action with state for interaction with environment """
        state = torch.tensor(np.reshape(state, [1, -1]), dtype=torch.float32)
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action = self.action_range * torch.tanh(
            mean + std * z
        )  # TanhNormal distribution as actions; re_parameterization trick

        action = self.action_range * torch.tanh(mean) if greedy else action
        return action.detach().numpy()[0]

    def sample_action(self):
        """ generate random actions for exploration """
        a = torch.Tensor(self.num_actions).uniform_(-1, 1)
        return self.action_range.numpy() * a.numpy()
