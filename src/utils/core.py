"""
Core training loop functions.
"""

from typing import Any, Dict
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import calculate_metrics
from src.wandb_utils import log_batch


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    log_frequency: int = 200,
) -> Dict[str, Any]:
    """
    Train the model for one epoch.
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Log to WandB
        if batch_idx % log_frequency == 0:
            log_batch(
                loss=loss.item(),
                learning_rate=lr_scheduler.get_last_lr()[0],
                epoch=epoch,
                batch_idx=batch_idx,
            )

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds, average="weighted")

    return {
        "loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "preds": all_preds,
        "labels": all_labels,
    }


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, Any]:
    """
    Validate one epoch.
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch}")

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_labels, all_preds, average="weighted")

    return {
        "loss": avg_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "preds": all_preds,
        "labels": all_labels,
    }


def train_single_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    log_frequency: int = 200,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train and validate for one epoch.
    
    Returns:
        Tuple of (train_metrics, val_metrics) dictionaries
    """
    train_metrics = train_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        epoch=epoch,
        log_frequency=log_frequency,
    )
    
    val_metrics = validate(
        model=model,
        dataloader=val_loader,
        device=device,
        epoch=epoch,
    )
    
    return train_metrics, val_metrics