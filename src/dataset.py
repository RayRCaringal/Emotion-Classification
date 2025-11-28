"""
FER2013 Dataset.
"""

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple, Any
import albumentations as A

class FER2013Dataset(Dataset):
    """
    FER2013 Dataset for emotion classification.
    """
    
    def __init__(self, split: str = "train", transform: Optional[A.Compose] = None):
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

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Returns
        -------
        Tuple[Any, int]
            Tuple containing:
            - image: image tensor
            - label: Emotion label (0-6)
        """
        item = self.dataset[idx]
        
        img = item["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        else:
            img = np.array(img.convert("RGB"))
        
        # Apply transforms
        if self.transform:
            img = self.transform(image=img)["image"]
        
        label = item["label"]
        return img, label


def get_datasets(augmentations_list: Optional[List[A.BasicTransform]] = None) -> Tuple[FER2013Dataset, FER2013Dataset, FER2013Dataset]:
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
    from src.transforms import get_transformations, base_transform
    
    train = FER2013Dataset(split="train", transform=get_transformations(augmentations_list))
    val = FER2013Dataset(split="valid", transform=base_transform())
    test = FER2013Dataset(split="test", transform=base_transform())

    return train, val, test