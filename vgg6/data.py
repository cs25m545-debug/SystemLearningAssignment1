import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from vgg6.config import Config

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


class Cutout:
    def __init__(self, length: int):
        self.length = length

    def __call__(self, image):
        if not self.length:
            return image
        height, width = image.shape[-2:]
        y, x = torch.randint(height, ()).item(), torch.randint(width, ()).item()
        half = self.length // 2
        image = image.clone()
        image[
            :,
            max(0, y - half) : min(height, y - half + self.length),
            max(0, x - half) : min(width, x - half + self.length),
        ] = 0
        return image


def image_transforms(cutout: int):
    evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    training = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            Cutout(cutout),
        ]
    )
    return training, evaluation


def split_indices(size: int, validation_size: int, seed: int):
    if not 0 < validation_size < size:
        raise ValueError("Validation split must leave nonempty training and validation sets")
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[validation_size:], indices[:validation_size]


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _loader(dataset, config: Config, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed),
        pin_memory=config.device == "cuda",
        persistent_workers=False,
    )


def _synthetic(size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.randn(size, 3, 32, 32, generator=generator),
        torch.randint(10, (size,), generator=generator),
    )


def training_loaders(config: Config):
    if config.synthetic:
        return (
            _loader(_synthetic(24, config.split_seed), config, True),
            _loader(_synthetic(12, config.split_seed + 1), config, False),
        )
    train_transform, evaluation_transform = image_transforms(config.cutout)
    augmented = datasets.CIFAR10(
        config.data_dir, train=True, download=True, transform=train_transform
    )
    clean = datasets.CIFAR10(
        config.data_dir, train=True, download=True, transform=evaluation_transform
    )
    train_indices, validation_indices = split_indices(
        len(augmented), config.validation_size, config.split_seed
    )
    return (
        _loader(Subset(augmented, train_indices), config, True),
        _loader(Subset(clean, validation_indices), config, False),
    )


def evaluation_loader(config: Config, split: str):
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if config.synthetic:
        offset = 1 if split == "validation" else 2
        return _loader(_synthetic(12, config.split_seed + offset), config, False)
    _, transform = image_transforms(config.cutout)
    dataset = datasets.CIFAR10(
        config.data_dir, train=split == "validation", download=True, transform=transform
    )
    if split == "validation":
        _, indices = split_indices(len(dataset), config.validation_size, config.split_seed)
        dataset = Subset(dataset, indices)
    return _loader(dataset, config, False)
