"""
Training functions for linear probe (frozen encoder, trainable classifier).
"""

from pathlib import Path

import torch
import wandb
from transformers import get_scheduler

from src.backup import BackupManager
from src.checkpoint_utils import save_checkpoint
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
from src.training_utils import (
    create_dataloaders,
    init_training_history,
    train_epoch,
    update_history,
    validate,
)
from src.wandb_utils import (
    cleanup_wandb_run,
    init_wandb_run,
    log_epoch,
)


def freeze_encoder(model: torch.nn.Module) -> torch.nn.Module:
    """
    Freeze Encoder Parameters 
    """

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise AttributeError(
            "Model does not have a 'classifier' attribute. "
            "Ensure you're using ViTForImageClassification."
        )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

    return model


def train_linear_probe(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    num_epochs: int = DEFAULT_LINEAR_PROBE_NUM_EPOCHS,
    batch_size: int = DEFAULT_LINEAR_PROBE_BATCH_SIZE,
    device: torch.device = DEVICE,
    save_path: Path = None,
    model_name: str = "linear_probe",
    use_wandb: bool = True,
    wandb_config: dict = None,
) -> tuple[torch.nn.Module, dict, Path]:
    """
    Train model with linear probe 
    """
    print("Freezing encoder parameters...")
    model = freeze_encoder(model)

    # Create run folder FIRST to get unique name
    run_folder = get_next_folder(f"{model_name}")
    unique_run_name = run_folder.name

    training_parameters = {
        "model_name": model_name,
        "model_architecture": type(model).__name__,
        "training_type": "linear_probe",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": DEFAULT_LINEAR_PROBE_LEARNING_RATE,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "device": str(device),
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "optimizer": type(optimizer).__name__,
        "optimizer_params": {
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0),
            "betas": optimizer.param_groups[0].get("betas", (0.9, 0.999)),
        },
        "run_folder": unique_run_name,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_params": sum(p.numel() for p in model.parameters()),
    }

    if use_wandb:
        if wandb_config is None:
            wandb_config = {
                "learning_rate": DEFAULT_LINEAR_PROBE_LEARNING_RATE,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "model_name": model_name,
                "architecture": type(model).__name__,
                "run_folder": unique_run_name,
                "training_type": "linear_probe",
            }
        else:
            wandb_config["run_folder"] = unique_run_name
            wandb_config["training_type"] = "linear_probe"

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

    # Learning Scheduler
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

    print(f"Training {model_name} for {num_epochs} epochs...")
    print(f"Training type: LINEAR PROBE (frozen encoder)")
    print(f"Run name (WandB): {unique_run_name}")
    print(f"Total training steps: {num_training_steps}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Run folder: {run_folder}")
    print(f"Best model will be saved to: {save_path}")
    print(f"Backup interval: every {backup_manager.backup_interval} epochs")

    if use_wandb and wandb.run is not None:
        print(f"W&B tracking: {wandb.run.url}")
    print("=" * 70)

    # Training loop
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
                save_checkpoint(
                    save_path=save_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_acc=val_acc,
                    val_loss=val_loss,
                    val_f1=val_f1,
                    history=history,
                    training_type="linear_probe",
                )
                print(
                    f"✅ New best model saved! (Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f})"
                )

            # Create backup checkpoint
            if backup_manager.should_backup(epoch):
                backup_path = backup_manager.create_backup(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    val_acc=val_acc,
                    val_loss=val_loss,
                )
                print(f"Backup created: {backup_path}")

    except Exception as e:
        print(f"\n❌ Training interrupted with error: {e}")
        print("Saving current state before exit...")

        # Save emergency checkpoint
        emergency_path = run_folder / f"emergency_checkpoint_{unique_run_name}.pth"
        torch.save(
            {
                "epoch": epoch if "epoch" in locals() else 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "training_type": "linear_probe",
            },
            emergency_path,
        )
        print(f"Emergency checkpoint saved to: {emergency_path}")

        if use_wandb:
            cleanup_wandb_run()

        raise

    print("\n" + "=" * 70)
    print("✅ Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    training_parameters["final_metrics"] = {
        "best_val_accuracy": best_val_acc,
        "final_val_f1": history["val_f1"][-1],
        "final_train_accuracy": history["train_acc"][-1],
    }

    # Save History
    save_training_parameters(run_folder, training_parameters)

    # Save history using metadata utility
    from src.metadata import save_training_history

    save_training_history(run_folder, history)

    # Create final backup
    final_backup_path = backup_manager.create_backup(
        model=model,
        optimizer=optimizer,
        epoch=num_epochs - 1,
        history=history,
        val_acc=val_acc,
        val_loss=val_loss,
        is_final=True,
    )
    print(f"Final backup created: {final_backup_path}")

    # SUCCESSFUL TRAINING COMPLETION - DELETE ALL BACKUPS
    print("Training completed successfully - cleaning up all backups...")
    deleted_count = backup_manager.cleanup_all_backups()
    print(f"Deleted {deleted_count} backup files")

    # Remove the now-empty backups folder
    folder_deleted = backup_manager.cleanup_backup_folder()
    if folder_deleted:
        print("Backups folder successfully removed")
    else:
        print("Backups folder could not be removed")

    # Finish W&B run with cleanup
    if use_wandb:
        cleanup_wandb_run()

    return model, history, run_folder