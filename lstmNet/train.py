import torch
from torch.utils.data import DataLoader

from .LSTM import LSTM
from . import cfg
from utils.MyDataset import MyDataset


def loss_fn(labels, outputs):
    summ = torch.abs(labels) + torch.abs(outputs) + cfg.epsilon
    smape = torch.mean(torch.abs(outputs - labels) / summ * 2.0)
    return smape


def train(data, data_type):
    dataset = MyDataset(data, 'predict1')
    data_loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = LSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_function = torch.nn.MSELoss()

    single_loss = None

    for i in range(cfg.epochs):
        for inputs, labels, mins, stds in data_loader:

            optimizer.zero_grad()
            inputs = torch.transpose(inputs, 1, 0)
            model.hidden_cell = (torch.zeros(1, cfg.batch_size, model.hidden_layer_size),
                                 torch.zeros(1, cfg.batch_size, model.hidden_layer_size))

            y_pred = model(inputs)
            y_pred = y_pred * stds + mins
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        # if i % 1 == 0:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        if i > 0 and i % 50 == 0:
            torch.save(model, '%s/lstm_%s.pt' % (cfg.model_path, data_type))
            print('save model success!')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
