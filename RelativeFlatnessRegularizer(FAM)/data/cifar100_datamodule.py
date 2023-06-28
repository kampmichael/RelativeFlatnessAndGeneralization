import torch.utils.data as dt
import torchvision.transforms as transforms
import torchvision.datasets as tdatasets

import pytorch_lightning as pl

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.cifar100_train = tdatasets.CIFAR100(self.data_dir, download=True, train=True, transform=self.train_transform)
            self.cifar100_val = tdatasets.CIFAR100(self.data_dir, download=True, train=False, transform=self.test_transform)
        if stage == 'test' or stage is None:
            self.cifar100_test = tdatasets.CIFAR100(self.data_dir, train=False, transform=self.test_transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        cifar100_train = dt.DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True)
        return cifar100_train

    def train_saving_dataloader(self):
        mnist_saving_train = dt.DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=False)
        return mnist_saving_train

    def val_dataloader(self):
        cifar100_val = dt.DataLoader(self.cifar100_val, batch_size=self.batch_size)
        return cifar100_val

    def test_dataloader(self):
        cifar100_test = dt.DataLoader(self.cifar100_test, batch_size=10 * self.batch_size)
        return cifar100_test