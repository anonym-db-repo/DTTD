import torch


class SoftQNet(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(SoftQNet, self).__init__()
        input_dim = num_inputs + num_actions

        negative_slope = 0.1
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=hidden_dim),
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

    def forward(self, q_input):
        x = self.linear1(q_input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
