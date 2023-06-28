import torch.utils.data as dt
import torchvision.transforms as transforms
import torchvision.datasets as tdatasets

import pytorch_lightning as pl

from util import Cutout
n_holes = 1
length = 16

class SVHNDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN),
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            Cutout(n_holes=n_holes, length=length),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            trainset = tdatasets.SVHN(root=self.data_dir, split='train', download=True, transform=self.train_transform)
            #extraset = tdatasets.SVHN(root=self.data_dir, split='extra', download=True, transform=self.train_transform)
            #self.svhn_train = dt.ConcatDataset([trainset, extraset])
            self.svhn_train = trainset
            self.svhn_val = tdatasets.SVHN(self.data_dir, download=True, split='test', transform=self.test_transform)
        if stage == 'test' or stage is None:
            self.svhn_test = tdatasets.SVHN(self.data_dir, split='test', transform=self.test_transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        svhn_train = dt.DataLoader(self.svhn_train, batch_size=self.batch_size, shuffle=True)
        return svhn_train

    def val_dataloader(self):
        svhn_val = dt.DataLoader(self.svhn_val, batch_size=self.batch_size)
        return svhn_val

    def test_dataloader(self):
        svhn_test = dt.DataLoader(self.svhn_test, batch_size=self.batch_size)
        return svhn_test