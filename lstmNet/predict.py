import torch
import numpy as np

from . import cfg


def test(init_input, data_type):  # [10, 3]
    y_preds = []

    init_inputs = np.zeros([cfg.batch_size, cfg.seq_lens, cfg.feature_size], dtype=np.float32)
    init_inputs[0, ...] = init_input
    lstm = torch.load('%s/lstm_%s.pt' % (cfg.model_path, data_type))

    for i in range(cfg.seq_lens):
        if i == 0:
            inputs = init_inputs

            data_min = np.min(inputs, axis=-2)
            data_std = np.max(inputs, axis=-2) - data_min
            inputs[:, 0, :] = (inputs[:, 0, :] - data_min) / data_std
            inputs = torch.from_numpy(inputs)
            inputs = torch.transpose(inputs, 1, 0)
            y_pred = lstm(inputs)
            y_preds.append(y_pred.detach().numpy() * data_std + data_min)

        else:
            concat_preds = np.array(y_preds)
            concat_preds = np.transpose(concat_preds, [1, 0, 2])
            inputs = np.concatenate([init_inputs[:, i:, :], concat_preds], axis=1)

            data_min = np.min(inputs, axis=-2)
            data_std = np.max(inputs, axis=-2) - data_min
            inputs[:, 0, :] = (inputs[:, 0, :] - data_min) / data_std

            inputs = torch.from_numpy(inputs)
            inputs = torch.transpose(inputs, 1, 0)
            y_pred = lstm(inputs)  # [100, 32, 3]
            y_preds.append(y_pred.detach().numpy() * data_std + data_min)

    return np.array(y_preds)[:, 0, :]


def test_1pos(init_input, data_type):
    data_min = np.min(init_input, axis=-2)
    data_std = np.max(init_input, axis=-2) - data_min
    init_input = (init_input - data_min) / data_std

    inputs = np.zeros([cfg.batch_size, cfg.seq_lens, cfg.feature_size], dtype=np.float32)
    inputs[0, ...] = init_input

    lstm = torch.load('%s/lstm_%s.pt' % (cfg.model_path, data_type))

    inputs = torch.from_numpy(inputs)
    inputs = torch.transpose(inputs, 1, 0)
    y_pred = lstm(inputs)

    y_pred = y_pred.detach().numpy()[0]
    y_pred = y_pred * data_std + data_min

    return y_pred.reshape(cfg.feature_size)
