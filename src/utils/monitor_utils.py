from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.utils.training_monitor import MonitorConfig, TrainingMonitor
from src.utils.train_config import (
    create_training_parameters,
    create_wandb_config,
    print_training_config,
    get_param_counts,
)
from src.utils.train_data import create_dataloaders
from src.utils.train_history import init_training_history, update_history
from src.utils.core import train_single_epoch
from src.wandb_utils import init_wandb_run, log_epoch, cleanup_wandb_run
from src.metadata import save_training_parameters, save_training_history
from src.config import DEVICE, NUM_WORKERS, PIN_MEMORY, DEFAULT_WARMUP_STEPS


def setup_training(
    model: torch.nn.Module,
    model_name: str,
    training_type: str,
    optimizer: torch.optim.Optimizer,
    train_dataset,
    val_dataset,
    num_epochs: int,
    batch_size: int,
    run_folder: Path,
    device: Optional[torch.device] = None,
    use_wandb: bool = True,
    wandb_config: Optional[Dict[str, Any]] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
) -> Tuple[TrainingMonitor, Dict[str, Any], DataLoader, DataLoader, dict, torch.optim.lr_scheduler._LRScheduler]:
    """
    Returns:
        Tuple of (monitor, training_params, train_loader, val_loader, history, lr_scheduler)
    """
    device = device or DEVICE
    
    monitor_config = MonitorConfig(
        run_folder=run_folder,
        model_name=model_name,
        max_backups=3,
        backup_interval=max(1, num_epochs // 4),
        save_best=True,
        save_final=True,
        save_emergency=True,
        emergency_run_name=f"{run_folder.name}_emergency",
    )
    monitor = TrainingMonitor(monitor_config)
    
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size, num_workers, pin_memory
    )
    
    # Create training parameters
    training_params = create_training_parameters(
        model=model,
        model_name=model_name,
        training_type=training_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        additional_params=additional_params,
    )
    
    save_training_parameters(run_folder, training_params)
    
    # Setup W&B
    wandb_run = None
    if use_wandb:
        wandb_config = create_wandb_config(
            training_parameters=training_params,
            training_type=training_type,
            run_name=run_folder.name,
            custom_config=wandb_config,
        )
        
        wandb_run = init_wandb_run(
            project="emotion-classification",
            name=run_folder.name,
            config=wandb_config,
            model=model,
        )
    
    # Setup learning rate scheduler
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=DEFAULT_WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )
    

    history = init_training_history()
    wandb_url = wandb_run.url if wandb_run else None
    param_counts = get_param_counts(model)
    
    print_training_config(
        model_name=type(model).__name__,
        training_type=training_type,
        param_counts=param_counts,
        run_name=run_folder.name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        train_loader_len=len(train_loader),
        val_loader_len=len(val_loader),
        run_folder=run_folder,
        save_path=run_folder / f"best_{model_name}.pth",
        backup_interval=monitor_config.backup_interval,
        wandb_url=wandb_url,
    )
    
    return monitor, training_params, train_loader, val_loader, history, lr_scheduler


def run_training(
    epoch: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    history: Dict[str, list],
    monitor: TrainingMonitor,
    use_wandb: bool = True,
    log_frequency: int = 200,
    cleanup_memory: bool = False,
    on_new_best_model: Optional[callable] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Run one training epoch.
    
    Returns:
        Tuple of (train_metrics, val_metrics, best_val_acc)
    """
    if cleanup_memory:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    train_metrics, val_metrics = train_single_epoch(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        epoch=epoch,
        log_frequency=log_frequency,
    )
    
    update_history(history, train_metrics, val_metrics)
    
    monitor.register_epoch_update(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        val_acc=val_metrics["accuracy"],
        val_loss=val_metrics["loss"],
        val_f1=val_metrics["f1"],
        history=history,
        lr_scheduler=lr_scheduler,
    )
    
    # Log to W&B
    if use_wandb:
        log_epoch(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
            val_f1=val_metrics["f1"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            learning_rate=lr_scheduler.get_last_lr()[0],
        )
    
    # Get best accuracy directly
    best_val_acc = max(history.get("val_acc", [0.0]))
    
    # Check if this is new best and call function if provided
    if val_metrics["accuracy"] >= best_val_acc and on_new_best_model:
        on_new_best_model(
            epoch=epoch,
            model=model,
            val_acc=val_metrics["accuracy"],
            run_folder=monitor.run_folder,
        )
    
    return train_metrics, val_metrics, best_val_acc


def finalize_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: Dict[str, list],
    training_parameters: Dict[str, Any],
    monitor: TrainingMonitor,
    best_val_acc: float,
    use_wandb: bool = True,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:

    # Update training parameters with final metrics
    training_parameters["final_metrics"] = {
        "best_val_accuracy": best_val_acc,
        "final_val_f1": history["val_f1"][-1] if history.get("val_f1") else 0.0,
        "final_train_accuracy": history["train_acc"][-1] if history.get("train_acc") else 0.0,
    }
    
    # Saving
    save_training_parameters(monitor.run_folder, training_parameters)
    save_training_history(monitor.run_folder, history)
    monitor.save_final_checkpoint(
        model=model,
        optimizer=optimizer,
        history=history,
        training_parameters=training_parameters,
    )
    
    # Clean Up
    monitor.cleanup_after_success()
    
    if use_wandb:
        cleanup_wandb_run()
    
    print(f"\n{'='*70}")
    print(f"✅ Training Completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*70}\n")