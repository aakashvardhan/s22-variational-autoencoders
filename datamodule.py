import lightning as L
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import datasets, transforms


class RGBMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.mnist = datasets.MNIST(root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        rgb_img = img.convert("RGB")

        if self.transform:
            rgb_img = self.transform(rgb_img)

        return rgb_img, label


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = ".", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.mnist_train = None
        self.mnist_test = None

    def prepare_data(self):
        # This method is called only once and on 1 GPU
        RGBMNISTDataset(self.data_dir, train=True, download=True)
        RGBMNISTDataset(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # This method is called on every GPU
        self.mnist_train = RGBMNISTDataset(
            self.data_dir, train=True, transform=self.transform
        )
        self.mnist_test = RGBMNISTDataset(
            self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.cifar_train = None
        self.cifar_test = None

        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.num_classes = 10

    def prepare_data(self):
        # download
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar_train = datasets.CIFAR10(
                self.data_dir, train=True, transform=self.transform
            )
            self.cifar_test = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
