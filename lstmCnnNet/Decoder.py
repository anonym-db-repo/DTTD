import torch
import torch.nn as nn

from . import cfg


class Decoder(nn.Module):
    def __init__(self, input_size=cfg.feature_size, hidden_layer_size=64):
        super(Decoder, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear_x = nn.Linear(hidden_layer_size, 1)
        self.linear_y = nn.Linear(hidden_layer_size, 1)
        self.linear_z = nn.Linear(hidden_layer_size, 1)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), -1, cfg.feature_size), self.hidden_cell)
        output_x = self.linear_x(lstm_out)
        output_y = self.linear_y(lstm_out)
        output_z = self.linear_z(lstm_out)
        return torch.cat((output_x, output_y, output_z), -1)
