"""
Checkpoint and state management utilities.
"""

from pathlib import Path

import torch

from .config import DEVICE


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer = None,
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"Val F1: {checkpoint.get('val_f1', 'N/A'):.4f}")

    return model, checkpoint


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_name: str = "google/vit-base-patch16-224-in21k",
    num_labels: int = 7,
) -> torch.nn.Module:
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"   Loaded model from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")

    return model


def save_checkpoint(
    save_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    val_loss: float,
    val_f1: float,
    history: dict,
) -> None:
    """
    Save a checkpoint of the training state.

    Parameters
    ----------
    save_path : Path
        Path to save checkpoint
    model : torch.nn.Module
        Model to save
    optimizer : torch.optim.Optimizer
        Optimizer to save
    epoch : int
        Current epoch
    val_acc : float
        Validation accuracy
    val_loss : float
        Validation loss
    val_f1 : float
        Validation F1 score
    history : dict
        Training history
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "history": history,
        },
        save_path,
    )
    print(f"Checkpoint saved: {save_path}")
