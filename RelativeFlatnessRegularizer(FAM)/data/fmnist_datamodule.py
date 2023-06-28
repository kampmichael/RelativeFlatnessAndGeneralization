import torch.utils.data as dt
import torchvision.transforms as transforms
import torchvision.datasets as tdatasets

import pytorch_lightning as pl

from util import Cutout
n_holes = 1
length = 16

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            #transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize(32),
            Cutout(n_holes=n_holes, length=length),
            transforms.Normalize((0.5,), (0.5,))
            ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize(32),
            transforms.Normalize((0.5,), (0.5,))
            ])

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.fmnist_train = tdatasets.FashionMNIST(self.data_dir, download=True, train=True, transform=self.train_transform)
            self.fmnist_val = tdatasets.FashionMNIST(self.data_dir, download=True, train=False, transform=self.test_transform)
        if stage == 'test' or stage is None:
            self.fmnist_test = tdatasets.FashionMNIST(self.data_dir, train=False, transform=self.test_transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        fmnist_train = dt.DataLoader(self.fmnist_train, batch_size=self.batch_size, shuffle=True)
        return fmnist_train

    def val_dataloader(self):
        fmnist_val = dt.DataLoader(self.fmnist_val, batch_size=self.batch_size)
        return fmnist_val

    def test_dataloader(self):
        fmnist_test = dt.DataLoader(self.fmnist_test, batch_size=self.batch_size)
        return fmnist_test