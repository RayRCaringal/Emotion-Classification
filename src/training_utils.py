"""
Common training utilities shared between full fine-tuning and linear probe.
"""

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
) -> tuple[float, float, float, float, float, list, list]:
    """
    Train the model for one epoch.
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

    return (
        avg_loss,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        all_preds,
        all_labels,
    )


def validate(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device, epoch: int
) -> tuple[float, float, float, float, float, list, list]:
    """
    Validate one epoch.
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

    return (
        avg_loss,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        all_preds,
        all_labels,
    )


def create_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def init_training_history() -> dict:
    return {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }


def update_history(
    history: dict,
    train_loss: float,
    train_acc: float,
    train_precision: float,
    train_recall: float,
    train_f1: float,
    val_loss: float,
    val_acc: float,
    val_precision: float,
    val_recall: float,
    val_f1: float,
) -> None:
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["train_precision"].append(train_precision)
    history["train_recall"].append(train_recall)
    history["train_f1"].append(train_f1)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_precision"].append(val_precision)
    history["val_recall"].append(val_recall)
    history["val_f1"].append(val_f1)