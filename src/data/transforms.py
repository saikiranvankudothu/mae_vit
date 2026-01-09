"""Data augmentation and transformation utilities."""

import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Optional, Tuple


class AlbumentationsTransform:
    """Wrapper for Albumentations transforms to work with PIL Images."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        """Apply albumentations transform to PIL Image."""
        # Convert PIL to numpy
        img_np = np.array(img)

        # Apply transform
        augmented = self.transform(image=img_np)
        img_tensor = augmented['image']

        return img_tensor


def get_mae_transforms(img_size: int = 224, mean: float = 0.485,
                      std: float = 0.229, is_train: bool = True):
    """
    Get transforms for MAE training.

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
        is_train: Whether for training (with augmentation)

    Returns:
        Albumentations transform
    """
    if is_train:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.Normalize(mean=[mean], std=[std]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[mean], std=[std]),
            ToTensorV2()
        ])

    return AlbumentationsTransform(transform)


def get_vit_transforms(img_size: int = 224, mean: float = 0.485,
                      std: float = 0.229, is_train: bool = True):
    """
    Get transforms for ViT training/inference.

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
        is_train: Whether for training (with augmentation)

    Returns:
        Albumentations transform
    """
    if is_train:
        transform = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
            ], p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Normalize(mean=[mean], std=[std]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[mean], std=[std]),
            ToTensorV2()
        ])

    return AlbumentationsTransform(transform)


def get_tta_transforms(img_size: int = 224, mean: float = 0.485,
                      std: float = 0.229, n_augmentations: int = 5):
    """
    Get Test-Time Augmentation transforms.

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
        n_augmentations: Number of TTA variations

    Returns:
        List of transforms
    """
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[mean], std=[std]),
        ToTensorV2()
    ])

    tta_transforms = [
        AlbumentationsTransform(base_transform),  # Original
    ]

    if n_augmentations >= 2:
        # Horizontal flip
        tta_transforms.append(AlbumentationsTransform(A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[mean], std=[std]),
            ToTensorV2()
        ])))

    if n_augmentations >= 3:
        # Slight rotation
        tta_transforms.append(AlbumentationsTransform(A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=5, p=1.0),
            A.Normalize(mean=[mean], std=[std]),
            ToTensorV2()
        ])))

    if n_augmentations >= 4:
        # Brightness adjustment
        tta_transforms.append(AlbumentationsTransform(A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[mean], std=[std]),
            ToTensorV2()
        ])))

    if n_augmentations >= 5:
        # Center crop
        tta_transforms.append(AlbumentationsTransform(A.Compose([
            A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=[mean], std=[std]),
            ToTensorV2()
        ])))

    return tta_transforms[:n_augmentations]


class AddNoise:
    """Add various types of noise to simulate low-quality images."""

    def __init__(self, noise_type: str = 'gaussian', intensity: float = 0.1):
        """
        Args:
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'poisson')
            intensity: Noise intensity
        """
        self.noise_type = noise_type
        self.intensity = intensity

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Add noise to image tensor."""
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(img) * self.intensity
            noisy_img = img + noise
        elif self.noise_type == 'salt_pepper':
            mask = torch.rand_like(img)
            noisy_img = img.clone()
            noisy_img[mask < self.intensity / 2] = 0  # Salt
            noisy_img[mask > 1 - self.intensity / 2] = 1  # Pepper
        elif self.noise_type == 'poisson':
            noisy_img = torch.poisson(img * 255 / self.intensity) * self.intensity / 255
        else:
            noisy_img = img

        return torch.clamp(noisy_img, 0, 1)


if __name__ == "__main__":
    print("Testing transforms...")

    # Test MAE transforms
    train_transform = get_mae_transforms(224, is_train=True)
    val_transform = get_mae_transforms(224, is_train=False)
    print("MAE transforms created successfully!")

    # Test ViT transforms
    train_transform = get_vit_transforms(224, is_train=True)
    val_transform = get_vit_transforms(224, is_train=False)
    print("ViT transforms created successfully!")

    # Test TTA
    tta_transforms = get_tta_transforms(224, n_augmentations=5)
    print(f"Created {len(tta_transforms)} TTA transforms!")

    print("\nTransforms module test completed!")
