import torch
import numpy as np

from . import cfg


def test(data, data_type):
    data_min = np.min(data, axis=-2)
    data_std = np.max(data, axis=-2) - data_min
    data = (data - data_min) / data_std

    inputs = np.zeros([cfg.batch_size, cfg.seq_lens, cfg.feature_size], dtype=np.float32)
    inputs[0] = data
    inputs = torch.from_numpy(inputs)
    inputs = torch.transpose(inputs, 1, 2)
    cnnLstm = torch.load('%s/cnnlstm_%s.pt' % (cfg.model_path, data_type))
    y_predicts = cnnLstm(inputs, None, cfg.seq_lens, 0.)
    y_pred = y_predicts[:, 0, :].detach().numpy()
    y_pred = y_pred * data_std + data_min
    return y_pred
