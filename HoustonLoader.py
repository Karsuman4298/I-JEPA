from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


def _load_mat_array(path, key=None):
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            arrays = {}

            def collect(name, value):
                if isinstance(value, h5py.Dataset):
                    arrays[name] = np.array(value)

            handle.visititems(collect)
    except (OSError, IOError):
        values = loadmat(path)
        arrays = {name: value for name, value in values.items() if not name.startswith("_")}

    if key is not None:
        matches = [value for name, value in arrays.items() if name == key or name.endswith("/" + key)]
        if not matches:
            raise KeyError(f"Key {key!r} was not found in {path}; available keys: {', '.join(arrays)}")
        return matches[0]
    if len(arrays) != 1:
        raise ValueError(f"Specify a MAT key for {path}; available keys: {', '.join(arrays)}")
    return next(iter(arrays.values()))


def load_houston_cube(image_path, label_path, image_key=None, label_key=None):
    cube = np.asarray(_load_mat_array(image_path, image_key), dtype=np.float32)
    labels = np.asarray(_load_mat_array(label_path, label_key)).squeeze()
    if cube.ndim != 3 or labels.ndim != 2:
        raise ValueError(f"Expected cube and labels with 3 and 2 dimensions, got {cube.shape} and {labels.shape}")
    if cube.shape[:2] != labels.shape:
        band_axes = [axis for axis, size in enumerate(cube.shape) if tuple(np.delete(cube.shape, axis)) == labels.shape]
        if len(band_axes) != 1:
            raise ValueError(f"Could not align cube {cube.shape} with labels {labels.shape}")
        cube = np.moveaxis(cube, band_axes[0], -1)
    flat_cube = cube.reshape(-1, cube.shape[-1])
    cube = (cube - flat_cube.mean(axis=0)) / np.maximum(flat_cube.std(axis=0), 1e-6)
    return cube.astype(np.float32), labels.astype(np.int64)


def stratified_positions(labels, train_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_positions = []
    test_positions = []
    for class_id in np.unique(labels[labels > 0]):
        positions = np.argwhere(labels == class_id)
        rng.shuffle(positions)
        train_count = int(len(positions) * train_fraction)
        train_positions.append(positions[:train_count])
        test_positions.append(positions[train_count:])
    return np.concatenate(train_positions), np.concatenate(test_positions)


class HoustonWindowDataset(Dataset):
    def __init__(self, cube, labels, positions, window_size=224):
        if window_size % 16:
            raise ValueError("window_size must be divisible by 16")
        radius = window_size // 2
        self.cube = torch.from_numpy(np.pad(
            cube, ((radius, radius), (radius, radius), (0, 0)), mode="reflect"
        )).permute(2, 0, 1).contiguous()
        self.labels = torch.from_numpy(labels)
        self.positions = torch.from_numpy(positions.astype(np.int64))
        self.radius = radius

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        y, x = self.positions[index].tolist()
        y += self.radius
        x += self.radius
        window = self.cube[:, y - self.radius:y + self.radius, x - self.radius:x + self.radius]
        return window, self.labels[y - self.radius, x - self.radius] - 1


def make_split_datasets(image_path, label_path, image_key=None, label_key=None,
                        window_size=224, train_fraction=0.2, seed=42):
    cube, labels = load_houston_cube(image_path, label_path, image_key, label_key)
    train_positions, test_positions = stratified_positions(labels, train_fraction, seed)
    return (
        HoustonWindowDataset(cube, labels, train_positions, window_size),
        HoustonWindowDataset(cube, labels, test_positions, window_size),
        labels,
    )


def make_loaders(train_dataset, test_dataset, batch_size=16, num_workers=0, seed=42):
    generator = torch.Generator().manual_seed(seed)
    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **options),
        DataLoader(test_dataset, shuffle=False, **options),
    )


class HoustonPrecomputedDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = torch.from_numpy(patches).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.num_bands = self.patches.shape[1]

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, index):
        return self.patches[index], self.labels[index]


def load_houston_patches(image_path, label_path, image_key=None, label_key=None):
    patches = np.asarray(_load_mat_array(image_path, image_key), dtype=np.float32)
    labels = np.asarray(_load_mat_array(label_path, label_key)).squeeze()
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if patches.ndim != 4:
        raise ValueError(f"Expected 4-D patch data, got {patches.shape}")
    sample_axes = [axis for axis, size in enumerate(patches.shape) if size == len(labels)]
    if len(sample_axes) != 1:
        raise ValueError(f"Could not identify sample axis in patches {patches.shape} for {len(labels)} labels")
    patches = np.moveaxis(patches, sample_axes[0], 0)
    band_axis = 1 + int(np.argmax(patches.shape[1:]))
    patches = np.moveaxis(patches, band_axis, -1)
    if patches.shape[1] != patches.shape[2]:
        raise ValueError(f"Expected square spatial patches, got {patches.shape}")
    flat = patches.reshape(-1, patches.shape[-1])
    patches = (patches - flat.mean(axis=0)) / np.maximum(flat.std(axis=0), 1e-6)
    labels = labels.astype(np.int64)
    if labels.min() == 1:
        labels = labels - 1
    return patches.astype(np.float32), labels


def make_precomputed_datasets(train_image, train_label, test_image, test_label,
                              train_image_key=None, train_label_key=None,
                              test_image_key=None, test_label_key=None):
    train_patches, train_labels = load_houston_patches(
        train_image, train_label, train_image_key, train_label_key
    )
    test_patches, test_labels = load_houston_patches(
        test_image, test_label, test_image_key, test_label_key
    )
    if train_patches.shape[1:] != test_patches.shape[1:]:
        raise ValueError(f"Train/test patch shapes differ: {train_patches.shape} and {test_patches.shape}")
    return HoustonPrecomputedDataset(train_patches, train_labels), HoustonPrecomputedDataset(test_patches, test_labels)
