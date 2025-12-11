"""
LoRA (Low-Rank Adaptation) training functions for Vision Transformer fine-tuning.
"""

import gc
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel
from transformers import get_scheduler
import wandb
from src.evaluate import evaluate_model
from src.backup import BackupManager
from src.checkpoint_utils import (
    safe_save_checkpoint,
    safe_load_checkpoint,
    save_model_checkpoint,
)
from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_WARMUP_STEPS,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
)
from src.metadata import get_next_folder, save_training_parameters
from src.models.base_model import BaseModel
from src.utils.training_utils import (
    train_epoch,
    validate,
    init_training_history,
    update_history,
    create_dataloaders,
        create_training_parameters,
    create_wandb_config,
    print_training_config,
    save_emergency_checkpoint,
    finalize_training,
)

def save_lora_for_inference(
    lora_model: PeftModel,
    run_folder: Path,
    base_model_name: str,
    num_labels: int
) -> dict:    
    # Clean up memory before saving
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lora_adapter_path = run_folder / "lora_adapter"
    full_checkpoint_path = run_folder / "best_lora_model.pth"
    merged_model_path = run_folder / "merged_model"

    # 1. Save LoRA adapters
    try:
        lora_model.save_pretrained(lora_adapter_path)
        print(f"✅ LoRA adapters saved to: {lora_adapter_path.name}")
    except Exception as e:
        print(f"❌ Failed to save LoRA adapters: {e}")
        lora_adapter_path = None

    # 2. Save full checkpoint with metadata
    try:
        # Get LoRA config
        lora_config_dict = {}
        if hasattr(lora_model, "peft_config") and "default" in lora_model.peft_config:
            lora_config_dict = lora_model.peft_config["default"].to_dict()

        checkpoint_data = {
            "model_state_dict": lora_model.state_dict(),
            "lora_config": lora_config_dict,
            "base_model_name": base_model_name,
            "num_labels": num_labels,
            "training_type": "lora",
        }

        success = safe_save_checkpoint(checkpoint_data, full_checkpoint_path)
        if not success:
            print(f"❌ Failed to save full checkpoint")
            full_checkpoint_path = None

    except Exception as e:
        print(f"❌ Error saving full checkpoint: {e}")
        full_checkpoint_path = None

    # 3. Try to save merged model (optional)
    try:
        print("🔀 Merging and saving model...")
        merged_model = lora_model.merge_and_unload()
        merged_model.save_pretrained(merged_model_path)
        print(f"✅ Merged model saved to: {merged_model_path.name}")

        del merged_model
        gc.collect()

    except Exception as e:
        print(f"⚠️ Could not save merged model: {e}")
        print("   (Not critical - you can still use LoRA adapters)")
        merged_model_path = None

    return {
        "lora_adapters": lora_adapter_path,
        "full_checkpoint": full_checkpoint_path,
        "merged_model": merged_model_path,
    }


def train_lora_model(
    base_model: BaseModel,
    lora_config: Optional[LoraConfig],
    optimizer: torch.optim.Optimizer,
    train_dataset,
    val_dataset,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = DEVICE,
    model_name: str = "lora_model",
    use_wandb: bool = True,
    wandb_config: Optional[dict] = None,
    gradient_accumulation_steps: int = 1,
):
    """
    Train a model using LoRA.
    
    Returns
    -------
    tuple
        (lora_model, history, run_folder)
    """
    from src.wandb_utils import cleanup_wandb_run, init_wandb_run, log_epoch

    # Setup LoRA on the base model
    print(f"🔧 Setting up LoRA on {base_model.model_name}...")
    lora_model = base_model.setup_for_lora(lora_config)
    
    # Get model metadata for saving
    base_model_name = base_model.model_name
    num_labels = base_model.num_labels

    # Create run folder
    run_folder = get_next_folder(f"{model_name}")
    unique_run_name = run_folder.name

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())

    # Create training parameters using helper
    training_parameters = create_training_parameters(
        model=lora_model,
        model_name=model_name,
        training_type="lora",
        num_epochs=num_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        num_workers=min(NUM_WORKERS, 4),
        pin_memory=PIN_MEMORY,
        additional_params={
            "base_model_name": base_model_name,
            "num_labels": num_labels,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "warmup_steps": DEFAULT_WARMUP_STEPS,
            "run_folder": unique_run_name,
            "lora_config": lora_config.to_dict() if lora_config else {},
        }
    )

    # Initialize W&B if requested
    if use_wandb:
        # Create base W&B config
        wandb_config = create_wandb_config(
            training_parameters=training_parameters,
            training_type="lora",
            run_name=unique_run_name,
            custom_config=wandb_config,
        )
        
        # Add LoRA-specific config if available
        if hasattr(lora_model, "peft_config") and "default" in lora_model.peft_config:
            config = lora_model.peft_config["default"]
            wandb_config.update({
                "lora_r": config.r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
            })

        init_wandb_run(
            project="emotion-classification",
            name=unique_run_name,
            config=wandb_config,
            model=lora_model,
        )

    # Create data loaders using shared utility
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=min(NUM_WORKERS, 4),
        pin_memory=PIN_MEMORY,
    )

    # Create learning rate scheduler
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
    lora_model.to(device)

    # Initialize training history using shared utility
    history = init_training_history()

    best_val_acc = 0.0
    best_model_path = run_folder / f"best_{model_name}.pth"

    # Save initial training parameters
    save_training_parameters(run_folder, training_parameters)

    # Setup backup manager
    backup_manager = BackupManager(
        run_folder=run_folder,
        max_backups=3,
        backup_interval=max(1, num_epochs // 4),
    )

    # Print training config using helper
    wandb_url = wandb.run.url if use_wandb and wandb.run is not None else None
    print_training_config(
        model_name=type(lora_model).__name__,
        training_type="lora",
        param_counts=type('obj', (object,), {
            'total': total_params,
            'trainable': trainable_params,
            'percentage_trainable': (trainable_params / total_params) * 100
        }),
        run_name=unique_run_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        train_loader_len=len(train_loader),
        val_loader_len=len(val_loader),
        run_folder=run_folder,
        save_path=best_model_path,
        backup_interval=backup_manager.backup_interval,
        wandb_url=wandb_url,
    )

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Clear cache at start of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Train using shared train_epoch utility
            (
                train_loss,
                train_acc,
                train_precision,
                train_recall,
                train_f1,
                train_preds,
                train_labels,
            ) = train_epoch(
                model=lora_model,
                dataloader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                epoch=epoch,
                log_frequency=50,  # Less frequent for LoRA
            )

            # Validate using shared validate utility
            (
                val_loss,
                val_acc,
                val_precision,
                val_recall,
                val_f1,
                val_preds,
                val_labels,
            ) = validate(
                model=lora_model,
                dataloader=val_loader,
                device=device,
                epoch=epoch,
            )

            # Update history using shared utility
            update_history(
                history=history,
                train_loss=train_loss,
                train_acc=train_acc,
                train_precision=train_precision,
                train_recall=train_recall,
                train_f1=train_f1,
                val_loss=val_loss,
                val_acc=val_acc,
                val_precision=val_precision,
                val_recall=val_recall,
                val_f1=val_f1,
            )

            # Log to W&B
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

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                # Clear memory before saving
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                # Save LoRA components for inference
                save_lora_for_inference(
                    lora_model=lora_model,
                    run_folder=run_folder,
                    base_model_name=base_model_name,
                    num_labels=num_labels
                )

                # Save training checkpoint using consolidated function
                success = save_model_checkpoint(
                    model=base_model,  # Use base_model for metadata
                    save_path=best_model_path,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_acc=val_acc,
                    val_loss=val_loss,
                    val_f1=val_f1,
                    history=history,
                    additional_data={
                        "training_parameters": training_parameters,
                        "best_val_acc": best_val_acc,
                        "lora_model_state": lora_model.state_dict(),
                    },
                )
                
                if success:
                    print(f"✅ Best model checkpoint saved")

            # Create backup if needed
            if backup_manager.should_backup(epoch):
                print("📦 Creating backup...")
                backup_path = backup_manager.create_backup(
                    model=lora_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    val_acc=val_acc,
                    val_loss=val_loss,
                )
                if backup_path:
                    print(f"✅ Backup created: {backup_path.name}")

            # Clear memory between epochs
            gc.collect()

    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user at epoch {epoch if 'epoch' in locals() else 'unknown'}")

    except Exception as e:
        print(f"\n❌ Training interrupted with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always save emergency checkpoint using helper
        save_emergency_checkpoint(
            model=lora_model,
            optimizer=optimizer,
            epoch=epoch if "epoch" in locals() else 0,
            history=history,
            training_parameters=training_parameters,
            run_folder=run_folder,
            run_name=unique_run_name,
        )

        if use_wandb:
            cleanup_wandb_run()

    # Finalize training using helper
    finalize_training(
        model=lora_model,
        optimizer=optimizer,
        history=history,
        training_parameters=training_parameters,
        run_folder=run_folder,
        backup_manager=backup_manager,
        best_val_acc=best_val_acc,
        use_wandb=use_wandb,
    )

    return lora_model, history, run_folder


def load_lora_model_for_inference(
    run_folder: Path,
    base_model: BaseModel,
    device: torch.device = DEVICE,
    load_method: str = "auto"
) -> torch.nn.Module:
    """
    Load a trained LoRA model for inference.
    
    Parameters
    ----------
    load_method : str
        Loading strategy: "auto", "merged", or "lora"
        
    """
    print(f" Loading LoRA model from: {run_folder.name}")

    lora_adapter_path = run_folder / "lora_adapter"
    full_checkpoint_path = run_folder / "best_lora_model.pth"
    merged_model_path = run_folder / "merged_model"

    # 1. Try merged model first (fastest for inference)
    if load_method in ["auto", "merged"] and merged_model_path.exists():
        print("🔀 Loading merged model...")
        try:
            from transformers import ViTForImageClassification
            
            model = ViTForImageClassification.from_pretrained(str(merged_model_path))
            model.to(device)
            model.eval()
            print(" Loaded merged model successfully")
            return model
        except Exception as e:
            print(f" Failed to load merged model: {e}")
            if load_method == "merged":
                raise

    # 2. Try LoRA adapter loading and merging
    if load_method in ["auto", "lora"] and lora_adapter_path.exists():
        print("🔧 Loading LoRA adapters and merging...")

        try:
            # Get the underlying model for PEFT operations
            underlying_model = base_model.get_underlying_model()
            
            # Load LoRA adapters onto base model
            lora_model = PeftModel.from_pretrained(underlying_model, lora_adapter_path)
            
            # Merge adapters into base weights
            merged_model = lora_model.merge_and_unload()
            
            # Load full state dict if checkpoint exists
            checkpoint = safe_load_checkpoint(full_checkpoint_path, device="cpu", weights_only=False)
            if checkpoint and "model_state_dict" in checkpoint:
                try:
                    merged_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    print("✅ Loaded classifier weights from checkpoint")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load full state dict: {e}")

            merged_model.to(device)
            merged_model.eval()
            print("✅ Loaded and merged LoRA model successfully")
            return merged_model

        except Exception as e:
            print(f"❌ Failed to load LoRA model: {e}")
            if load_method == "lora":
                raise

    raise FileNotFoundError(
        f"No valid model files found in {run_folder} for load_method='{load_method}'"
    )
