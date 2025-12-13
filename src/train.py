"""
Training functions for full fine-tuning.
"""

from pathlib import Path
from typing import Optional
import torch

from src.models.base_model import BaseModel
from src.metadata import get_next_folder
from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
)
from src.utils.monitor_utils import (
    setup_training,
    run_training,
    finalize_training,
)


def train_model(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    train_dataset,
    val_dataset,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None,
    model_name: str = "model",
    use_wandb: bool = True,
    wandb_config: Optional[dict] = None,
) -> tuple[BaseModel, dict, Path]:

    print("Configuring model for Full Fine-Tuning...")
    model.setup_for_full_finetune()
    run_folder = get_next_folder(f"{model_name}_full")

    monitor, training_params, train_loader, val_loader, history, lr_scheduler = setup_training(
        model=model,
        model_name=model_name,
        training_type="full_fine_tuning",
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        run_folder=run_folder,
        device=device,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
        additional_params={
            "base_model_name": model.model_name,
            "num_labels": model.num_labels,
            "learning_rate": DEFAULT_LEARNING_RATE,
        }
    )
    

    model.to(device)
    best_val_acc = 0.0
    
    try:
        for epoch in range(num_epochs):
            train_metrics, val_metrics, best_val_acc = run_training(
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=model.device,
                history=history,
                monitor=monitor,
                use_wandb=use_wandb,
            )
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        finalize_training(
            model=model,
            optimizer=optimizer,
            history=history,
            training_parameters=training_params,
            monitor=monitor,
            best_val_acc=best_val_acc,
            use_wandb=use_wandb,
            lr_scheduler=lr_scheduler,
        )
        
    except KeyboardInterrupt:
        print(f"\n Training interrupted")
        monitor.save_emergency_checkpoint(
            model=model,
            optimizer=optimizer,
            history=history,
            training_parameters=training_params,
            lr_scheduler=lr_scheduler,
        )
        raise
        
    except Exception as e:
        print(f"\n Training error: {e}")
        monitor.save_emergency_checkpoint(
            model=model,
            optimizer=optimizer,
            history=history,
            training_parameters=training_params,
            lr_scheduler=lr_scheduler,
        )
        raise
    
    return model, history, run_folder