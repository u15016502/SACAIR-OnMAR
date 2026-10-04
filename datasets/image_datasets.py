"""
Image Dataset Handling

This module handles loading and preprocessing of image datasets used in the CNN configuration application.
Datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, ISIC Melanoma, Mosquito, FruitsGB
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import os


class ImageDatasetLoader:
    """
    Handles loading and preprocessing of various image datasets.
    """

    def __init__(self, dataset_name: str, data_dir: str = "./data", random_seed: int = 42):
        """
        Initialize the dataset loader.

        Args:
            dataset_name: Name of the dataset to load
            data_dir: Directory to store/load datasets
            random_seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Dataset configurations
        self.dataset_configs = {
            'mnist': {
                'num_classes': 10,
                'input_shape': (1, 28, 28),
                'mean': (0.1307,),
                'std': (0.3081,)
            },
            'fashion-mnist': {
                'num_classes': 10,
                'input_shape': (1, 28, 28),
                'mean': (0.5,),
                'std': (0.5,)
            },
            'cifar-10': {
                'num_classes': 10,
                'input_shape': (3, 32, 32),
                'mean': (0.4914, 0.4822, 0.4465),
                'std': (0.2023, 0.1994, 0.2010)
            },
            'cifar-100': {
                'num_classes': 100,
                'input_shape': (3, 32, 32),
                'mean': (0.5071, 0.4867, 0.4408),
                'std': (0.2675, 0.2565, 0.2761)
            }
        }

    def get_transforms(self, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Get data transforms for training and testing.

        Args:
            augment: Whether to apply data augmentation to training data

        Returns:
            Tuple of (train_transform, test_transform)
        """
        config = self.dataset_configs.get(self.dataset_name)
        if config is None:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Base transforms
        base_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ]

        # Training transforms with augmentation
        if augment and self.dataset_name in ['cifar-10', 'cifar-100']:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                *base_transforms
            ])
        else:
            train_transform = transforms.Compose(base_transforms)

        # Test transforms (no augmentation)
        test_transform = transforms.Compose(base_transforms)

        return train_transform, test_transform

    def load_dataset(self, val_split: float = 0.1, batch_size: int = 128, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load the dataset and create data loaders.

        Args:
            val_split: Fraction of training data to use for validation
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_transform, test_transform = self.get_transforms()

        # Load datasets based on name
        if self.dataset_name == 'mnist':
            train_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=train_transform
            )
            test_dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=test_transform
            )

        elif self.dataset_name == 'fashion-mnist':
            train_dataset = datasets.FashionMNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=train_transform
            )
            test_dataset = datasets.FashionMNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=test_transform
            )

        elif self.dataset_name == 'cifar-10':
            train_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=train_transform
            )
            test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=test_transform
            )

        elif self.dataset_name == 'cifar-100':
            train_dataset = datasets.CIFAR100(
                root=self.data_dir,
                train=True,
                download=True,
                transform=train_transform
            )
            test_dataset = datasets.CIFAR100(
                root=self.data_dir,
                train=False,
                download=True,
                transform=test_transform
            )

        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not yet implemented. "
                                    f"Supported: {list(self.dataset_configs.keys())}")

        # Split training data into train and validation
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )

        # Create data loaders.
        #
        # persistent_workers matters here: a run executes one timestep per
        # epoch and iterates these loaders dozens of times, and without it the
        # worker processes are torn down and respawned on every single pass.
        # prefetch_factor keeps batches queued so the device is not waiting on
        # the input pipeline.
        loader_options = {
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': num_workers > 0,
        }
        if num_workers > 0:
            loader_options['prefetch_factor'] = 4

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            **loader_options
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_options
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_options
        )

        return train_loader, val_loader, test_loader

    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        if self.dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        config = self.dataset_configs[self.dataset_name]
        return {
            'name': self.dataset_name,
            'num_classes': config['num_classes'],
            'input_shape': config['input_shape'],
            'channels': config['input_shape'][0],
            'height': config['input_shape'][1],
            'width': config['input_shape'][2]
        }
