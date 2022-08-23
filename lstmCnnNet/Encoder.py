import torch
import torch.nn as nn

from . import cfg


class Encoder(nn.Module):
    def __init__(self, input_size=cfg.feature_size, hidden_layer_size=64, output_size=cfg.feature_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.conv1 = torch.nn.Conv1d(input_size, 16, 1, bias=True, padding=0, padding_mode='zeros')
        self.conv2 = torch.nn.Conv1d(input_size, 32, 3, bias=True, padding=1, padding_mode='zeros')
        self.conv3 = torch.nn.Conv1d(input_size, 16, 5, bias=True, padding=2, padding_mode='zeros')
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size)
        self.hidden_cell = (torch.zeros(1, cfg.batch_size, self.hidden_layer_size),
                            torch.zeros(1, cfg.batch_size, self.hidden_layer_size))
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, inputs):
        feature1 = self.conv1(inputs)
        feature3 = self.conv2(inputs)
        feature5 = self.conv3(inputs)
        feature = torch.cat((feature1, feature3, feature5), 1)
        feature = feature.permute(2, 0, 1)
        lstm_output, self.hidden_cell = self.lstm(feature, self.hidden_cell)
        output = self.linear(lstm_output[-1, ...].view(-1, self.hidden_layer_size))
        return output

