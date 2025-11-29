import albumentations as A
from albumentations.pytorch import ToTensorV2


def base_transform():
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_transformations(augmentations_list=None):
    """Get training transforms with optional augmentations"""
    transforms = []

    # Add custom augmentations if provided
    if augmentations_list:
        transforms.extend(augmentations_list)

    base = base_transform()
    transforms.extend(base.transforms)

    return A.Compose(transforms)
