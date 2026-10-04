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
        train_dataset, test_dataset = self._build_datasets(train_transform, test_transform)

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

    def _build_datasets(self, train_transform, test_transform) -> Tuple[Dataset, Dataset]:
        """
        Construct the torchvision train and test datasets, downloading if needed.

        Args:
            train_transform: Transform applied to the training split
            test_transform: Transform applied to the test split

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
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

        return train_dataset, test_dataset

    def load_arrays(
        self,
        num_train: int,
        num_test: int = 0,
        scale_255: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a fixed sample of the dataset as raw numpy arrays.

        The clustering composition application needs the data as arrays rather
        than as loaders: it clusters one fixed sample of instances and carries
        those clusters across timesteps, so it must hold the same instances
        throughout rather than iterate fresh batches.

        No augmentation and no normalisation is applied. That is deliberate -
        the pretrained feature extractors used by the clustering design space
        are Keras models that apply their own ``preprocess_input``, which
        expects raw 0-255 pixels; handing them already-normalised tensors
        puts the input off the scale the donor network was trained on.

        Args:
            num_train: Number of training instances to sample
            num_test: Number of held-out test instances to sample
            scale_255: Return pixels on 0-255 rather than 0-1

        Returns:
            Tuple of (train_images, train_labels, test_images, test_labels),
            images shaped (N, C, H, W)
        """
        to_tensor = transforms.Compose([transforms.ToTensor()])
        train_dataset, test_dataset = self._build_datasets(to_tensor, to_tensor)

        def take(dataset, count: int) -> Tuple[np.ndarray, np.ndarray]:
            if count <= 0:
                return np.empty((0,)), np.empty((0,))

            count = min(count, len(dataset))
            # A fixed shuffled prefix, so the sample is class-balanced in
            # expectation rather than being whatever order the files are in -
            # CIFAR's test split is not shuffled on disk.
            generator = torch.Generator().manual_seed(self.random_seed)
            indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()

            images = []
            labels = []
            for index in indices:
                image, label = dataset[index]
                images.append(image.numpy())
                labels.append(label)

            images = np.asarray(images, dtype=np.float32)
            if scale_255:
                images = images * 255.0

            return images, np.asarray(labels)

        train_images, train_labels = take(train_dataset, num_train)
        test_images, test_labels = take(test_dataset, num_test)

        return train_images, train_labels, test_images, test_labels

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
