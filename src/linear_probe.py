"""
Training functions for linear probe (frozen encoder, trainable classifier).
"""

from pathlib import Path
from typing import Optional, Any, Dict, Tuple

import torch
import wandb
from transformers import get_scheduler
from src.models.base_model import BaseModel

from src.backup import BackupManager
from src.checkpoint_utils import save_model_checkpoint, safe_save_checkpoint
from src.config import (
    DEFAULT_LINEAR_PROBE_BATCH_SIZE,
    DEFAULT_LINEAR_PROBE_LEARNING_RATE,
    DEFAULT_LINEAR_PROBE_NUM_EPOCHS,
    DEFAULT_WARMUP_STEPS,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
)
from src.metadata import get_next_folder, save_training_parameters

from src.wandb_utils import (
    cleanup_wandb_run,
    init_wandb_run,
    log_epoch,
)

from src.utils.training_utils import (
    create_training_parameters,
    create_wandb_config,
    print_training_config,
    save_emergency_checkpoint,
    finalize_training,
    create_dataloaders,
    init_training_history,
    train_epoch,
    update_history,
    validate,
)


def train_linear_probe(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    num_epochs: int = DEFAULT_LINEAR_PROBE_NUM_EPOCHS,
    batch_size: int = DEFAULT_LINEAR_PROBE_BATCH_SIZE,
    device: torch.device = DEVICE,
    save_path: Optional[Path] = None,
    model_name: str = "linear_probe",
    use_wandb: bool = True,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> Tuple[BaseModel, dict, Path]: 
  
    print("🔧 Configuring model for Linear Probe...")
    model.setup_for_linear_probe()
    param_counts = model.count_parameters()
    print(f"   Trainable: {param_counts.trainable:,} / {param_counts.total:,} ({param_counts.percentage_trainable:.2f}%)")
    
    # Create run folder
    run_folder = get_next_folder(f"{model_name}")
    unique_run_name = run_folder.name

    # Create training parameters using helper
    training_parameters = create_training_parameters(
        model=model,
        model_name=model_name,
        training_type="linear_probe",
        num_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        additional_params={
            "base_model_name": model.model_name,
            "num_labels": model.num_labels,
            "learning_rate": DEFAULT_LINEAR_PROBE_LEARNING_RATE,
            "warmup_steps": DEFAULT_WARMUP_STEPS,
            "run_folder": unique_run_name,
        }
    )

    # Initialize W&B if requested
    if use_wandb:
        wandb_config = create_wandb_config(
            training_parameters=training_parameters,
            training_type="linear_probe",
            run_name=unique_run_name,
            custom_config=wandb_config,
        )
        
        init_wandb_run(
            project="emotion-classification",
            name=unique_run_name,
            config=wandb_config,
            model=model,
        )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size, NUM_WORKERS, PIN_MEMORY
    )

    # Learning rate scheduler
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=DEFAULT_WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    training_parameters["lr_scheduler"] = {
        "type": "linear",
        "num_training_steps": num_training_steps,
        "num_warmup_steps": DEFAULT_WARMUP_STEPS,
    }

    # Move model to device
    model.to(device)

    # Training history
    history = init_training_history()

    best_val_acc = 0.0

    if save_path is None:
        save_path = run_folder / f"best_{model_name}.pth"

    save_training_parameters(run_folder, training_parameters)

    # Initialize backup manager
    backup_manager = BackupManager(
        run_folder=run_folder,
        max_backups=3,
        backup_interval=max(1, num_epochs // 4),
    )

    wandb_url = wandb.run.url if use_wandb and wandb.run is not None else None
    print_training_config(
        model_name=type(model).__name__,
        training_type="linear_probe",
        param_counts=param_counts,
        run_name=unique_run_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        train_loader_len=len(train_loader),
        val_loader_len=len(val_loader),
        run_folder=run_folder,
        save_path=save_path,
        backup_interval=backup_manager.backup_interval,
        wandb_url=wandb_url,
    )

    # Training loop
    epoch = 0 
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)

            # Train
            (
                train_loss,
                train_acc,
                train_precision,
                train_recall,
                train_f1,
                train_preds,
                train_labels,
            ) = train_epoch(model, train_loader, optimizer, lr_scheduler, device, epoch)

            # Validate
            (
                val_loss,
                val_acc,
                val_precision,
                val_recall,
                val_f1,
                val_preds,
                val_labels,
            ) = validate(model, val_loader, device, epoch)

            # Save history
            update_history(
                history,
                train_loss,
                train_acc,
                train_precision,
                train_recall,
                train_f1,
                val_loss,
                val_acc,
                val_precision,
                val_recall,
                val_f1,
            )

            # Log to WandB
            if use_wandb:
                log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    val_f1=val_f1,
                    val_precision=val_precision,
                    val_recall=val_recall,
                    learning_rate=lr_scheduler.get_last_lr()[0],
                )

            print(
                f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}"
            )
            print(
                f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}"
            )

            # Save Best Model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                success = save_model_checkpoint(
                    model=model,
                    save_path=save_path,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_acc=val_acc,
                    val_loss=val_loss,
                    val_f1=val_f1,
                    history=history,
                    additional_data={
                        "training_parameters": training_parameters,
                        "best_val_acc": best_val_acc,
                        "lr_scheduler_state": lr_scheduler.state_dict(),
                    },
                )
                
                if success:
                    print(
                        f" New best model saved! (Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f})"
                    )
                else:
                    print(
                        f" Failed to save best model checkpoint"
                    )

            # Create backup checkpoint
            if backup_manager.should_backup(epoch):
                backup_path = backup_manager.create_backup(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    history=history,
                    val_acc=val_acc,
                    val_loss=val_loss,
                )
                if backup_path:
                    print(f" Backup created: {backup_path.name}")

    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user at epoch {epoch + 1}")
        print("Saving current state before exit...")

    except Exception as e:
        print(f"\n❌ Training interrupted with error: {e}")
        print("Saving emergency checkpoint...")
        import traceback
        traceback.print_exc()

    finally:
        # Always save emergency checkpoint on interruption
        if epoch < num_epochs - 1:  # Only if training didn't complete
            save_emergency_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                training_parameters=training_parameters,
                run_folder=run_folder,
                run_name=unique_run_name,
                lr_scheduler=lr_scheduler,
            )

        # Always cleanup W&B if needed
        if use_wandb:
            cleanup_wandb_run()

    # Finalize training using helper
    finalize_training(
        model=model,
        optimizer=optimizer,
        history=history,
        training_parameters=training_parameters,
        run_folder=run_folder,
        backup_manager=backup_manager,
        best_val_acc=best_val_acc,
        use_wandb=use_wandb,
        lr_scheduler=lr_scheduler,
    )

    return model, history, run_folder