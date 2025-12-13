"""
LoRA (Low-Rank Adaptation) training functions for Vision Transformer fine-tuning - simplified.
"""

import gc
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from torch.optim import AdamW
from peft import LoraConfig, PeftModel
from src.checkpoint_utils import safe_save_checkpoint, safe_load_checkpoint
from src.models.base_model import BaseModel
from src.metadata import get_next_folder
from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    NUM_WORKERS,
)
from src.utils.monitor_utils import (
    setup_training,
    run_training,
    finalize_training,
)


def save_lora_for_inference(
    lora_model: PeftModel,
    run_folder: Path,
    base_model_name: str,
    num_labels: int
) -> Dict[str, Optional[Path]]:
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

    # Save full checkpoint with metadata
    try:

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
            print(f" Failed to save full checkpoint")
            full_checkpoint_path = None

    except Exception as e:
        print(f" Error saving full checkpoint: {e}")
        full_checkpoint_path = None

    try:
        print(" Merging & Saving model...")
        merged_model = lora_model.merge_and_unload()
        merged_model.save_pretrained(merged_model_path)
        print(f" Saved to: {merged_model_path.name}")

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
    train_dataset,
    val_dataset,
    learning_rate: float,
    weight_decay: float = 0.01,
    lora_config: LoraConfig = None,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[torch.device] = None,
    model_name: str = "lora_model",
    use_wandb: bool = True,
    wandb_config: Optional[dict] = None,
    gradient_accumulation_steps: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: List[str] = None,
) -> tuple[PeftModel, dict, Path]:
    """
    Train a model using LoRA.
    """

    print(f"🔧 Setting up LoRA on {base_model.model_name}...")
    print(f"   LoRA Config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
          f"dropout={lora_config.lora_dropout}")
    print(f"   Target modules: {lora_config.target_modules}")
    
    lora_model = base_model.setup_for_lora(lora_config)
    
    optimizer = AdamW(
        lora_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())
    
    print(f" Optimizer created with {trainable_params:,} trainable parameters "
          f"({(trainable_params/total_params)*100:.2f}% of total)")
    
    run_folder = get_next_folder(f"{model_name}_lora")
    
    def save_lora_adapters(epoch, model, val_acc, run_folder):
        save_lora_for_inference(
            lora_model=model,
            run_folder=run_folder,
            base_model_name=base_model.model_name,
            num_labels=base_model.num_labels
        )
    
    monitor, training_params, train_loader, val_loader, history, lr_scheduler = setup_training(
        model=lora_model,
        model_name=model_name,
        training_type="lora",
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
            "base_model_name": base_model.model_name,
            "num_labels": base_model.num_labels,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "percentage_trainable": (trainable_params / total_params) * 100,
            "lora_config": lora_config.to_dict() if lora_config else {},
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules,
        },
        num_workers=min(4, NUM_WORKERS),  # Use fewer workers for LoRA
    )
    
    if device:
        lora_model.to(device)
    
    best_val_acc = 0.0
    
    try:
        for epoch in range(num_epochs):
            train_metrics, val_metrics, best_val_acc = run_training(
                epoch=epoch,
                model=lora_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=lora_model.device,
                history=history,
                monitor=monitor,
                use_wandb=use_wandb,
                log_frequency=50,  
                cleanup_memory=True,  
                on_new_best_model=save_lora_adapters, 
            )
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Clean memory between epochs
            gc.collect()
        
        finalize_training(
            model=lora_model,
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
            model=lora_model,
            optimizer=optimizer,
            history=history,
            training_parameters=training_params,
            lr_scheduler=lr_scheduler,
        )
        raise
        
    except Exception as e:
        print(f"\n Training error: {e}")
        monitor.save_emergency_checkpoint(
            model=lora_model,
            optimizer=optimizer,
            history=history,
            training_parameters=training_params,
            lr_scheduler=lr_scheduler,
        )
        raise
    
    return lora_model, history, run_folder


def load_lora_model_for_inference(
    run_folder: Path,
    base_model: BaseModel,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    load_method: str = "auto"
) -> torch.nn.Module:
    """
    Load a trained LoRA model for inference.
    """
    print(f"Loading LoRA model from: {run_folder.name}")

    lora_adapter_path = run_folder / "lora_adapter"
    full_checkpoint_path = run_folder / "best_lora_model.pth"
    merged_model_path = run_folder / "merged_model"

    # 1. Try merged model first (fastest for inference)
    if load_method in ["auto", "merged"] and merged_model_path.exists():
        print(" Loading merged model...")
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
            lora_model = PeftModel.from_pretrained(underlying_model, lora_adapter_path)
            
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
            print("Loaded and merged LoRA model successfully")
            return merged_model

        except Exception as e:
            print(f"Failed to load LoRA model: {e}")
            if load_method == "lora":
                raise

    raise FileNotFoundError(
        f"No valid model files found in {run_folder} for load_method='{load_method}'"
    )