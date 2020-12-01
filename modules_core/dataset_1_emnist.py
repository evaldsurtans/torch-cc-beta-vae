import torch.utils
import torch
import torchvision
import torch.random

class DataSet(torch.utils.data.Dataset):

    def __init__(self, is_train):
        self.data = torchvision.datasets.EMNIST(
            root='./tmp',
            split='balanced',
            download=True,
            train=is_train
        )

    @property
    def classes(self):
        return self.data.classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.unsqueeze(self.data.data[idx].t(), dim=0) / 255.0

        noise = torch.rand(x.size())
        x_noisy = torch.where(noise < 0.2, torch.zeros_like(x), x)

        y = self.data.targets[idx]
        return x, x_noisy, y