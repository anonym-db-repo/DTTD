import random
import numpy as np
from torch.utils.data import Dataset
import torchvision


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
