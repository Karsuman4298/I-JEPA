
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

def get_mini_imagenet_loaders(
    data_dir: str | Path = "./imagenet",
    batch_size: int = 128,
    num_workers: int = 0,
    val_split: float = 0.1,
    pin_memory: bool = True,
    seed: int = 42,
):
    """
    Return train_loader, val_loader (or None), test_loader for Mini-ImageNet.

    Parameters
    ----------
    data_dir   : root folder containing class subfolders
    batch_size : how many images per batch
    num_workers: DataLoader subprocesses; try 0 on Windows
    val_split  : fraction of training set to reserve for validation
    pin_memory : speeds up host‑to‑GPU transfer (set False if CPU‑only)
    seed       : ensures deterministic train/val split
    """

    # Common ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Use ImageFolder for Mini-ImageNet
    full_ds = datasets.ImageFolder(root=data_dir, transform=train_tfms)

    # Split into train/val
    if val_split and 0 < val_split < 1.0:
        g = torch.Generator().manual_seed(seed)
        val_len = int(len(full_ds) * val_split)
        train_len = len(full_ds) - val_len
        train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=g)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory)
    else:
        train_ds = full_ds
        val_loader = None

    # For simplicity, reuse the same dataset for test (or point to another folder if you have one)
    test_ds = datasets.ImageFolder(root=data_dir, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, val_loader, test_loader