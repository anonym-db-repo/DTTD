import torch
from torch.utils.data import DataLoader

from .Encoder import Encoder
from .Decoder import Decoder
from .cnnLSTM import CnnLstm
from . import cfg
from utils.MyDataset import MyDataset


def loss_fn(outputs, labels):
    summ = torch.abs(labels) + torch.abs(outputs) + cfg.epsilon
    smape = torch.mean(torch.abs(outputs - labels) / summ * 2.0)
    return smape


def train(data, data_type):
    dataset = MyDataset(data, 'predict10')
    data_loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    encoder = Encoder()
    decoder = Decoder()
    model = CnnLstm(encoder, decoder, cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    single_loss = None

    for i in range(cfg.epochs):
        for inputs, labels, mins, stds in data_loader:
            optimizer.zero_grad()
            inputs = torch.transpose(inputs, 1, 2)
            labels = torch.transpose(labels, 1, 0)
            model.encoder.hidden_cell = (torch.zeros(1, cfg.batch_size, model.encoder.hidden_layer_size),
                                         torch.zeros(1, cfg.batch_size, model.encoder.hidden_layer_size))
            model.decoder.hidden_cell = (torch.zeros(1, 1, model.decoder.hidden_layer_size),
                                         torch.zeros(1, 1, model.decoder.hidden_layer_size))

            outputs = model(inputs, labels, labels.shape[0])
            outputs = outputs * stds + mins

            single_loss = loss_fn(outputs, labels)
            single_loss.backward()
            optimizer.step()

        # if i % 1 == 0:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        if i > 0 and i % 50 == 0:
            torch.save(model, '%s/cnnlstm_%s.pt' % (cfg.model_path, data_type))
            print('save model success!')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

