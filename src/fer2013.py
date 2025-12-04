"""
FER2013 Dataset.
"""

from typing import Any

import albumentations as A
import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset for emotion classification.

    Dataset Images are Single Channel Grayscale so we Conver them to 3 Channel RGB for ViT. Needs to be changed potentially if planning ot use with other models
    """

    def __init__(self, split: str = "train", transform: A.Compose | None = None):
        """

        Parameters
        ----------
        split : optional
            Dataset split to load, by default "train"
            Options: "train", "valid", "test"
        transform : optional
            Albumentations transforms to apply, by default None
        """
        self.dataset = load_dataset("AutumnQiu/fer2013", split=split)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        item = self.dataset[idx]

        img = item["image"]
        label = item["label"]

        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = np.array(img)

        # If grayscale, convert to 3 channels for ViT
        if len(img_array.shape) == 2:  # [H, W]
            img_array = np.stack([img_array] * 3, axis=-1)  # [H, W, 3]

        # Apply transforms
        if self.transform:
            img_array = self.transform(image=img_array)["image"]

        return img_array, label


def get_datasets(
    augmentations_list: list[A.BasicTransform] | None = None,
) -> tuple[FER2013Dataset, FER2013Dataset, FER2013Dataset]:
    """
    Parameters
    ----------
    augmentations_list : optional
        List of transforms to apply during training,

    Returns
    -------
        Tuple containing:
        - train_dataset
        - val_dataset
        - test_dataset
    """
    from src.transforms import base_transform, get_transformations

    train = FER2013Dataset(
        split="train", transform=get_transformations(augmentations_list)
    )
    val = FER2013Dataset(split="valid", transform=base_transform())
    test = FER2013Dataset(split="test", transform=base_transform())

    return train, val, test
