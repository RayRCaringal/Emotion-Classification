from pathlib import Path
from typing import Dict, Any, Optional
import torch


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

    if hasattr(model, "count_parameters"):
        param_counts = model.count_parameters()
  
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


def get_param_counts(model: torch.nn.Module) -> Any:
    if hasattr(model, 'count_parameters'):
        return model.count_parameters()
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return type('obj', (object,), {
        'total': total,
        'trainable': trainable,
        'percentage_trainable': (trainable / total * 100) if total > 0 else 0
    })