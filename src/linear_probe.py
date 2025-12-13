"""
Training functions for linear probe (frozen encoder, trainable classifier).
"""

from pathlib import Path
from typing import Optional, Any, Dict, Tuple
import torch

from src.models.base_model import BaseModel
from src.metadata import get_next_folder
from src.config import (
    DEFAULT_LINEAR_PROBE_BATCH_SIZE,
    DEFAULT_LINEAR_PROBE_LEARNING_RATE,
    DEFAULT_LINEAR_PROBE_NUM_EPOCHS,
)
from src.utils.monitor_utils import (
    setup_training,
    run_training,
    finalize_training,
)

def train_linear_probe(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    device: torch.device,
    num_epochs: int = DEFAULT_LINEAR_PROBE_NUM_EPOCHS,
    batch_size: int = DEFAULT_LINEAR_PROBE_BATCH_SIZE,
    save_path: Optional[Path] = None,
    model_name: str = "linear_probe",
    use_wandb: bool = True,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> Tuple[BaseModel, dict, Path]: 
  
    print("Configuring model for Linear Probe...")
    model.setup_for_linear_probe()
    
    run_folder = get_next_folder(f"{model_name}_linear")
    

    monitor, training_params, train_loader, val_loader, history, lr_scheduler = setup_training(
        model=model,
        model_name=model_name,
        training_type="linear_probe",
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
            "learning_rate": DEFAULT_LINEAR_PROBE_LEARNING_RATE,
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
        print(f"\n⚠️ Training interrupted")
        monitor.save_emergency_checkpoint(
            model=model,
            optimizer=optimizer,
            history=history,
            training_parameters=training_params,
            lr_scheduler=lr_scheduler,
        )
        raise
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        monitor.save_emergency_checkpoint(
            model=model,
            optimizer=optimizer,
            history=history,
            training_parameters=training_params,
            lr_scheduler=lr_scheduler,
        )
        raise
    
    return model, history, run_folder