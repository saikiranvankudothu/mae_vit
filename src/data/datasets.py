"""Dataset classes for NIH Chest X-ray, Shenzhen, and Montgomery TB datasets."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split


class NIHChestXrayDataset(Dataset):
    """
    NIH Chest X-ray Dataset for MAE pretraining.

    Dataset contains 112,120 frontal-view X-ray images from 30,805 patients.
    Used for unsupervised denoising pretraining.

    Args:
        root_dir (str): Root directory containing images
        csv_file (str): Path to Data_Entry_2017.csv
        transform (callable, optional): Transform to apply to images
        subset (str): 'train' or 'val'
        split_ratio (float): Train/val split ratio
    """
    def __init__(self, root_dir: str, csv_file: Optional[str] = None,
                 transform: Optional[Callable] = None, subset: str = 'train',
                 split_ratio: float = 0.9, seed: int = 42):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.subset = subset

        # Load image paths
        self.image_paths = []

        if csv_file and os.path.exists(csv_file):
            # Load from CSV
            df = pd.read_csv(csv_file)
            all_images = df['Image Index'].tolist()
        else:
            # Scan directory
            for img_dir in self.root_dir.glob('images*'):
                for img_path in img_dir.glob('*.png'):
                    all_images.append(img_path.name)

        # Train/val split
        train_imgs, val_imgs = train_test_split(
            all_images, train_size=split_ratio, random_state=seed
        )

        self.image_names = train_imgs if subset == 'train' else val_imgs

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_name = self.image_names[idx]

        # Find image in subdirectories
        img_path = None
        for img_dir in self.root_dir.glob('images*'):
            candidate = img_dir / img_name
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # Fallback
            img_path = self.root_dir / img_name

        # Load image
        image = Image.open(img_path).convert('L')  # Grayscale

        if self.transform:
            image = self.transform(image)

        return image


class TBChestXrayDataset(Dataset):
    """
    Combined Shenzhen and Montgomery TB Dataset for classification.

    Args:
        root_dirs (List[str]): List of root directories for datasets
        transform (callable, optional): Transform to apply
        subset (str): 'train', 'val', or 'test'
        train_ratio (float): Training data ratio
        val_ratio (float): Validation data ratio
        stratify (bool): Whether to stratify split by class
    """
    def __init__(self, root_dirs: List[str], transform: Optional[Callable] = None,
                 subset: str = 'train', train_ratio: float = 0.7,
                 val_ratio: float = 0.15, stratify: bool = True, seed: int = 42):
        self.transform = transform
        self.subset = subset

        # Collect all images and labels
        self.samples = []  # List of (img_path, label)

        for root_dir in root_dirs:
            root_path = Path(root_dir)

            # Shenzhen dataset structure
            if 'Shenzhen' in str(root_path) or 'shenzhen' in str(root_path):
                self._load_shenzhen(root_path)
            # Montgomery dataset structure
            elif 'Montgomery' in str(root_path) or 'montgomery' in str(root_path):
                self._load_montgomery(root_path)
            else:
                # Generic structure: root/TB and root/Normal
                self._load_generic(root_path)

        # Split into train/val/test
        self._create_splits(train_ratio, val_ratio, stratify, seed)

    def _load_shenzhen(self, root_path: Path):
        """Load Shenzhen dataset."""
        # TB cases
        tb_dir = root_path / 'TB'
        if tb_dir.exists():
            for img_path in tb_dir.glob('*.png'):
                self.samples.append((str(img_path), 1))  # TB = 1

        # Normal cases
        normal_dir = root_path / 'Normal'
        if normal_dir.exists():
            for img_path in normal_dir.glob('*.png'):
                self.samples.append((str(img_path), 0))  # Normal = 0

    def _load_montgomery(self, root_path: Path):
        """Load Montgomery dataset."""
        self._load_generic(root_path)

    def _load_generic(self, root_path: Path):
        """Load dataset with TB/Normal folder structure."""
        for label, folder_name in [(1, 'TB'), (0, 'Normal')]:
            folder_path = root_path / folder_name
            if folder_path.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in folder_path.glob(ext):
                        self.samples.append((str(img_path), label))

    def _create_splits(self, train_ratio: float, val_ratio: float,
                      stratify: bool, seed: int):
        """Create train/val/test splits."""
        test_ratio = 1.0 - train_ratio - val_ratio

        paths, labels = zip(*self.samples)
        paths, labels = list(paths), list(labels)

        if stratify:
            # Train split
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                paths, labels, train_size=train_ratio,
                stratify=labels, random_state=seed
            )

            # Val/Test split
            val_size = val_ratio / (val_ratio + test_ratio)
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels, train_size=val_size,
                stratify=temp_labels, random_state=seed
            )
        else:
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                paths, labels, train_size=train_ratio, random_state=seed
            )
            val_size = val_ratio / (val_ratio + test_ratio)
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels, train_size=val_size, random_state=seed
            )

        # Select subset
        if self.subset == 'train':
            self.samples = list(zip(train_paths, train_labels))
        elif self.subset == 'val':
            self.samples = list(zip(val_paths, val_labels))
        else:  # test
            self.samples = list(zip(test_paths, test_labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('L')  # Grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """Get class distribution in current subset."""
        labels = [label for _, label in self.samples]
        return {
            'Normal': labels.count(0),
            'TB': labels.count(1),
            'Total': len(labels)
        }


def get_mae_dataloaders(root_dir: str, csv_file: Optional[str] = None,
                       transform_train: Optional[Callable] = None,
                       transform_val: Optional[Callable] = None,
                       batch_size: int = 64, num_workers: int = 4,
                       split_ratio: float = 0.9, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for MAE training.

    Args:
        root_dir: Path to NIH dataset
        csv_file: Path to Data_Entry_2017.csv
        transform_train: Training transforms
        transform_val: Validation transforms
        batch_size: Batch size
        num_workers: Number of data loading workers
        split_ratio: Train/val split ratio
        seed: Random seed

    Returns:
        train_loader, val_loader
    """
    train_dataset = NIHChestXrayDataset(
        root_dir, csv_file, transform_train, 'train', split_ratio, seed
    )

    val_dataset = NIHChestXrayDataset(
        root_dir, csv_file, transform_val, 'val', split_ratio, seed
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"MAE Training: {len(train_dataset)} images")
    print(f"MAE Validation: {len(val_dataset)} images")

    return train_loader, val_loader


def get_vit_dataloaders(root_dirs: List[str],
                       transform_train: Optional[Callable] = None,
                       transform_val: Optional[Callable] = None,
                       batch_size: int = 32, num_workers: int = 4,
                       train_ratio: float = 0.7, val_ratio: float = 0.15,
                       stratify: bool = True, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for ViT training.

    Args:
        root_dirs: List of dataset root directories
        transform_train: Training transforms
        transform_val: Validation/test transforms
        batch_size: Batch size
        num_workers: Number of workers
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        stratify: Stratify splits by class
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = TBChestXrayDataset(
        root_dirs, transform_train, 'train',
        train_ratio, val_ratio, stratify, seed
    )

    val_dataset = TBChestXrayDataset(
        root_dirs, transform_val, 'val',
        train_ratio, val_ratio, stratify, seed
    )

    test_dataset = TBChestXrayDataset(
        root_dirs, transform_val, 'test',
        train_ratio, val_ratio, stratify, seed
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"ViT Training: {train_dataset.get_class_distribution()}")
    print(f"ViT Validation: {val_dataset.get_class_distribution()}")
    print(f"ViT Test: {test_dataset.get_class_distribution()}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing dataset classes...")

    # Test would require actual data
    print("Dataset module loaded successfully!")
