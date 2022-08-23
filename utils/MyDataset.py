from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data_all, data_type):
        super(MyDataset, self).__init__()

        ds, labels, mins, stds = [], [], [], []
        if data_type == 'predict1':
            for i in range(390):
                data = data_all[i: 10 + i, :]
                label = data_all[10 + i, :]
                data_min = np.min(data, axis=-2)
                data_std = np.max(data, axis=-2) - data_min
                data = (data - data_min) / data_std

                ds.append(data)
                labels.append(label)
                mins.append(data_min)
                stds.append(data_std)

        elif data_type == 'predict2':
            for i in range(380):
                data = data_all[i: 10 + i, :]
                label = np.zeros([2, 3])
                label[0] = data_all[10 + i, :]
                label[1] = data_all[19 + i, :]

                data_min = np.min(data, axis=-2)
                data_std = np.max(data, axis=-2) - data_min
                data = (data - data_min) / data_std

                ds.append(data)
                labels.append(label)
                mins.append(data_min)
                stds.append(data_std)

        elif data_type == 'predict10':
            for i in range(380):
                data = data_all[i: 10 + i, :]
                label = data_all[10 + i: 20 + i, :]
                data_min = np.min(data, axis=-2)
                data_std = np.max(data, axis=-2) - data_min
                data = (data - data_min) / data_std

                ds.append(data)
                labels.append(label)
                mins.append(data_min)
                stds.append(data_std)

        self.data = np.array(ds, dtype=np.float32)
        self.label = np.array(labels, dtype=np.float32)
        self.mins = np.array(mins, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.mins[index], self.stds[index]

    def __len__(self):
        return len(self.data)


test_path = './data/parabola_test_data_2.npy'
# test_path = './data/eight_test_data_2.npy'
# test_path = './data/circle_test_data_200.npy'
drone_test_data = np.load(test_path)
