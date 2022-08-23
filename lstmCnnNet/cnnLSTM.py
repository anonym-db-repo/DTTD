import torch
import torch.nn as nn
import random

from . import cfg


class CnnLstm(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target, target_length, teach_forcing_rate=0.5):
        encoder_output = self.encoder(input_seq)
        decoder_input = encoder_output.to(cfg.device)
        outputs = torch.zeros([target_length, cfg.batch_size, input_seq.shape[1]]).to(self.device)
        for t in range(target_length):
            decoder_output = self.decoder(decoder_input)
            outputs[t] = decoder_output.view(cfg.batch_size, cfg.feature_size)
            teach_forcing = random.random() < teach_forcing_rate
            decoder_input = target[t, ...] if teach_forcing else decoder_output
        return outputs
