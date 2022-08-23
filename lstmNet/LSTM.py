import torch
import torch.nn as nn
from . import cfg


class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=64, output_size=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        # self.lstm.mode = 'RNN_RELU'
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, cfg.batch_size, self.hidden_layer_size),
                            torch.zeros(1, cfg.batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), -1, 3), self.hidden_cell)
        predictions = self.linear(lstm_out[-1])
        return predictions
