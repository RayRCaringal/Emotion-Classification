"""
Training helper functions shared across full fine-tuning, linear probe, and LoRA training.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import calculate_metrics
from src.wandb_utils import log_batch



def create_training_parameters(
    model: torch.nn.Module,
    model_name: str,
    training_type: str,
    num_epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    train_dataset,
    val_dataset,
    device,
    num_workers: int,
    pin_memory: bool,
    additional_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Create standardized training parameters dictionary.
    """
    if hasattr(model, "count_parameters"):
        param_counts = model.count_parameters()
    else:
        # Fallback for non-BaseModel models
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts = type('obj', (object,), {
            'total': total,
            'trainable': trainable,
            'percentage_trainable': (trainable / total * 100) if total > 0 else 0
        })
    
    base_params = {
        "model_name": model_name,
        "model_architecture": type(model).__name__,
        "training_type": training_type,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "device": str(device),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "optimizer": type(optimizer).__name__,
        "optimizer_params": {
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
            "betas": optimizer.param_groups[0].get("betas", (0.9, 0.999)),
        },
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "total_parameters": param_counts.total,
        "trainable_parameters": param_counts.trainable,
        "trainable_percentage": param_counts.percentage_trainable,
    }
    
    if additional_params:
        base_params.update(additional_params)
    
    return base_params


def create_wandb_config(
    training_parameters: Dict[str, Any],
    training_type: str,
    run_name: str,
    custom_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Create standardized WandB configuration.
    """
    default_config = {
        "learning_rate": training_parameters["optimizer_params"]["lr"],
        "weight_decay": training_parameters["optimizer_params"]["weight_decay"],
        "batch_size": training_parameters["batch_size"],
        "epochs": training_parameters["num_epochs"],
        "model_name": training_parameters["model_name"],
        "architecture": training_parameters["model_architecture"],
        "training_type": training_type,
        "run_folder": run_name,
        "total_parameters": training_parameters["total_parameters"],
        "trainable_parameters": training_parameters["trainable_parameters"],
    }
    
    if custom_config:
        default_config.update(custom_config)
    
    return default_config


def print_training_config(
    model_name: str,
    training_type: str,
    param_counts,
    run_name: str,
    num_epochs: int,
    batch_size: int,
    device,
    train_loader_len: int,
    val_loader_len: int,
    run_folder: Path,
    save_path: Path,
    backup_interval: int,
    wandb_url: Optional[str] = None,
):
    """
    Print standardized training configuration.
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name} for {num_epochs} epochs")
    print(f"{'='*70}")
    print(f"Training type: {training_type.upper().replace('_', ' ')}")
    if hasattr(model_name, '__name__'):
        print(f"Model architecture: {model_name.__name__}")
    print(f"Trainable parameters: {param_counts.trainable:,} ({param_counts.percentage_trainable:.2f}%)")
    print(f"Total parameters: {param_counts.total:,}")
    print(f"Run name: {run_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {train_loader_len}")
    print(f"Val batches: {val_loader_len}")
    print(f"Run folder: {run_folder}")
    print(f"Best model save path: {save_path}")
    print(f"Backup interval: every {backup_interval} epochs")
    
    if wandb_url:
        print(f"W&B tracking: {wandb_url}")
    print(f"{'='*70}\n")


def save_emergency_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict,
    training_parameters: Dict,
    run_folder: Path,
    run_name: str,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """
    Save emergency checkpoint on training interruption.
    """
    emergency_path = run_folder / f"emergency_checkpoint_{run_name}.pth"
    
    emergency_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "training_parameters": training_parameters,
    }
    
    if lr_scheduler is not None:
        emergency_data["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    
    from src.checkpoint_utils import safe_save_checkpoint
    success = safe_save_checkpoint(emergency_data, emergency_path)
    
    if success:
        print(f"✅ Emergency checkpoint saved to: {emergency_path.name}")
    else:
        print(f"❌ Failed to save emergency checkpoint")
    
    return success


def finalize_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: Dict,
    training_parameters: Dict,
    run_folder: Path,
    backup_manager,
    best_val_acc: float,
    use_wandb: bool = True,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """
    Finalize training - save final state, cleanup, etc.
    """
    # Update training parameters with final metrics
    training_parameters["final_metrics"] = {
        "best_val_accuracy": best_val_acc,
        "final_val_f1": history["val_f1"][-1] if history["val_f1"] else 0.0,
        "final_train_accuracy": history["train_acc"][-1] if history["train_acc"] else 0.0,
    }
    
    # Save updated parameters
    from src.metadata import save_training_parameters
    save_training_parameters(run_folder, training_parameters)
    
    # Save training history
    from src.metadata import save_training_history
    save_training_history(run_folder, history)
    
    # Create final backup
    final_backup_path = backup_manager.create_backup(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch=training_parameters["num_epochs"] - 1,
        history=history,
        val_acc=best_val_acc,
        val_loss=history["val_loss"][-1] if history["val_loss"] else 0.0,
        is_final=True,
    )
    
    if final_backup_path:
        print(f"📦 Final backup created: {final_backup_path.name}")
    
    # Cleanup W&B
    if use_wandb:
        from src.wandb_utils import cleanup_wandb_run
        cleanup_wandb_run()
    
    # Clean up backups if training completed successfully
    if training_parameters.get("training_completed", True):
        print("\n🧹 Training completed successfully - cleaning up backups...")
        deleted_count = backup_manager.cleanup_all_backups()
        print(f"   Deleted {deleted_count} backup files")
        
        # Remove the now-empty backups folder
        folder_deleted = backup_manager.cleanup_backup_folder()
        if folder_deleted:
            print("   Backups folder removed")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("✅ Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*70}\n")


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