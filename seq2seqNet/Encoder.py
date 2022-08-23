import torch
import torch.nn as nn

from . import cfg


class Encoder(nn.Module):
    def __init__(self, input_size=cfg.feature_size, hidden_layer_size=64, output_size=cfg.feature_size):
        super(Encoder, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, cfg.batch_size, self.hidden_layer_size),
                            torch.zeros(1, cfg.batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), -1, cfg.feature_size), self.hidden_cell)
        output = self.linear(lstm_out[-1, :, :].view(cfg.batch_size, self.hidden_layer_size))
        return output
