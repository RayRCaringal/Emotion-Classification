"""
LoRA (Low-Rank Adaptation) training functions for Vision Transformer fine-tuning.
"""

import gc
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from src.backup import BackupManager
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


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
    bias: str = "none",
) -> LoraConfig:
    if target_modules is None:
        target_modules = ["query", "value"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )

    return config


def get_lora_model(
    base_model, lora_config: LoraConfig, unfreeze_classifier: bool = True
):
    """
    Apply LoRA configuration to base model and optionally unfreeze classifier head.
    """
    model = get_peft_model(base_model, lora_config)

    if unfreeze_classifier:
        # Unfreeze the classifier head for training
        classifier_names = ["classifier", "classifier_head", "head", "fc"]

        for name in classifier_names:
            try:
                # Try different possible locations for classifier
                if hasattr(model, name):
                    classifier = getattr(model, name)
                elif hasattr(model.base_model, name):
                    classifier = getattr(model.base_model, name)
                elif hasattr(model.base_model.model, name):
                    classifier = getattr(model.base_model.model, name)
                else:
                    continue

                if hasattr(classifier, "parameters"):
                    for param in classifier.parameters():
                        param.requires_grad = True
                    print(f"✅ Classifier head '{name}' unfrozen for training")
                    break
            except AttributeError:
                continue

        # If no classifier found, warn but continue
        if not any(param.requires_grad for param in model.parameters()):
            print(
                "⚠️ Warning: No parameters require gradient. Checking all parameters..."
            )
            for name, param in model.named_parameters():
                if "classifier" in name or "head" in name or "fc" in name:
                    param.requires_grad = True
                    print(f"   Unfroze: {name}")

    model.print_trainable_parameters()
    return model


def safe_save_checkpoint(state_dict, filepath, metadata=None):
    """Safely save checkpoint with proper error handling."""
    try:
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Use torch.save with _use_new_zipfile_serialization for better compatibility
        torch.save(state_dict, filepath, _use_new_zipfile_serialization=True)

        # Verify the file was written
        if filepath.exists():
            file_size = filepath.stat().st_size
            print(f"✅ Checkpoint saved: {filepath} ({file_size:,} bytes)")
            return True
        else:
            print(f"❌ Failed to save checkpoint: {filepath}")
            return False

    except Exception as e:
        print(f"❌ Error saving checkpoint {filepath}: {e}")
        return False


def safe_load_checkpoint(filepath, device="cpu", weights_only=False):
    """Safely load checkpoint with proper error handling."""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"❌ Checkpoint not found: {filepath}")
            return None

        file_size = filepath.stat().st_size
        print(f"📁 Loading checkpoint: {filepath} ({file_size:,} bytes)")

        # Try loading with weights_only flag for safety
        checkpoint = torch.load(
            filepath, map_location=device, weights_only=weights_only
        )

        if checkpoint is not None:
            print("✅ Checkpoint loaded successfully")
        else:
            print("❌ Checkpoint loaded but is None")

        return checkpoint

    except Exception as e:
        print(f"❌ Error loading checkpoint {filepath}: {e}")

        # Try alternative loading methods
        try:
            print("🔄 Trying alternative loading method...")
            import pickle

            with open(filepath, "rb") as f:
                checkpoint = pickle.load(f)
            print("✅ Loaded with pickle")
            return checkpoint
        except:
            print("❌ All loading methods failed")
            return None


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    device: torch.device,
    epoch: int,
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss

        # Check if loss requires grad
        if not loss.requires_grad:
            if batch_idx == 0:  # Only warn on first batch to avoid spam
                print(f"⚠️ WARNING: Loss doesn't require gradient at epoch {epoch}")
                trainable_params = [
                    (name, p.requires_grad)
                    for name, p in model.named_parameters()
                    if p.requires_grad
                ]
                if not trainable_params:
                    print("❌ CRITICAL: No parameters require gradients!")
                    # Try emergency fix
                    for name, param in model.named_parameters():
                        if "lora" in name.lower():
                            param.requires_grad = True
                    print("   Emergency: Set all LoRA parameters to require_grad=True")
                else:
                    print(
                        f"✅ Found {len(trainable_params)} parameters with gradients enabled"
                    )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Gradient clipping
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Clear cache periodically to prevent memory buildup
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1


def validate(model, dataloader: DataLoader, device: torch.device, epoch: int):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch}")

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1


def save_lora_model_for_inference(
    model: PeftModel, run_folder, model_name: str = "lora_model"
):
    """
    Save LoRA model for inference with robust error handling.
    """
    print("\nSaving model for inference...")

    # Clean up memory before saving
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lora_save_path = run_folder / "lora_adapter"
    best_model_path = run_folder / f"best_{model_name}.pth"

    try:
        # Save LoRA adapters
        model.save_pretrained(lora_save_path)
        print(f"✅ LoRA adapters saved to: {lora_save_path}")
    except Exception as e:
        print(f"❌ Failed to save LoRA adapters: {e}")
        lora_save_path = None

    try:
        # Save full model checkpoint
        base_model_name = getattr(
            model.base_model.config,
            "_name_or_path",
            "google/vit-base-patch16-224-in21k",
        )
        num_labels = getattr(model.base_model.config, "num_labels", 7)

        # Get LoRA config safely
        lora_config_dict = {}
        if hasattr(model, "peft_config") and "default" in model.peft_config:
            lora_config_dict = model.peft_config["default"].to_dict()

        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "lora_config": lora_config_dict,
            "base_model_name": base_model_name,
            "num_labels": num_labels,
        }

        # Save checkpoint
        success = safe_save_checkpoint(checkpoint_data, best_model_path)
        if success:
            print(f"✅ Full model checkpoint saved to: {best_model_path}")
        else:
            print(f"❌ Failed to save checkpoint: {best_model_path}")
            best_model_path = None

    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        best_model_path = None

    merged_save_path = None
    try:
        # Try to save merged model
        print("Merging and saving model...")
        merged_model = model.merge_and_unload()
        merged_save_path = run_folder / "merged_model"
        merged_model.save_pretrained(merged_save_path)
        print(f"✅ Merged model saved to: {merged_save_path}")

        # Clear merged model from memory
        del merged_model
        gc.collect()

    except Exception as e:
        print(f"⚠️ Could not save merged model: {e}")
        print("   This is not critical - you can still use LoRA adapters")
        merged_save_path = None

    return {
        "lora_adapters": lora_save_path,
        "full_checkpoint": best_model_path,
        "merged_model": merged_save_path,
    }


def train_lora_model(
    model,
    optimizer: torch.optim.Optimizer,
    train_dataset,
    val_dataset,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = DEVICE,
    model_name: str = "lora_model",
    use_wandb: bool = True,
    wandb_config: dict | None = None,
    gradient_accumulation_steps: int = 1,  # Add gradient accumulation
):
    from src.wandb_utils import cleanup_wandb_run, init_wandb_run, log_epoch

    run_folder = get_next_folder(f"{model_name}")
    unique_run_name = run_folder.name

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    training_parameters = {
        "model_name": model_name,
        "model_type": "LoRA",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "device": str(device),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": (trainable_params / total_params) * 100,
        "run_folder": unique_run_name,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
    }

    if use_wandb:
        if wandb_config is None:
            wandb_config = {
                "learning_rate": DEFAULT_LEARNING_RATE,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "model_name": model_name,
                "model_type": "LoRA",
                "lora_r": model.peft_config["default"].r
                if hasattr(model, "peft_config")
                else 8,
                "lora_alpha": model.peft_config["default"].lora_alpha
                if hasattr(model, "peft_config")
                else 16,
                "lora_dropout": model.peft_config["default"].lora_dropout
                if hasattr(model, "peft_config")
                else 0.1,
                "trainable_params": trainable_params,
                "run_folder": unique_run_name,
            }
        else:
            wandb_config["run_folder"] = unique_run_name
            wandb_config["model_type"] = "LoRA"
            wandb_config["trainable_params"] = trainable_params

        init_wandb_run(
            project="emotion-classification",
            name=unique_run_name,
            config=wandb_config,
            model=model,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(NUM_WORKERS, 4),  # Limit workers to prevent memory issues
        pin_memory=PIN_MEMORY,
        persistent_workers=False,  # Disable persistent workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(NUM_WORKERS, 4),
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
    )

    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=DEFAULT_WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    # Move model to device
    model.to(device)

    history = {
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

    best_val_acc = 0.0
    best_model_path = run_folder / f"best_{model_name}.pth"

    save_training_parameters(run_folder, training_parameters)

    backup_manager = BackupManager(
        run_folder=run_folder,
        max_backups=3,
        backup_interval=max(1, num_epochs // 4),
    )

    print(f"Training LoRA model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(
        f"Trainable parameters: {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}%)"
    )
    print(f"Run folder: {run_folder}")
    print(f"Device: {device}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Clear cache at start of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            train_loss, train_acc, train_precision, train_recall, train_f1 = (
                train_epoch(model, train_loader, optimizer, lr_scheduler, device, epoch)
            )

            val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
                model, val_loader, device, epoch
            )

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

            if val_acc > best_val_acc:
                best_val_acc = val_acc

                # Clear memory before saving
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                save_paths = save_lora_model_for_inference(
                    model=model, run_folder=run_folder, model_name=model_name
                )

                # Save training checkpoint
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "history": history,
                    "training_parameters": training_parameters,
                }

                success = safe_save_checkpoint(checkpoint_data, best_model_path)
                if success:
                    print(f"✅ Training checkpoint saved (Val Acc: {val_acc:.4f})")
                else:
                    print("❌ Failed to save training checkpoint")

            if backup_manager.should_backup(epoch):
                print("Creating backup...")
                backup_path = backup_manager.create_backup(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    val_acc=val_acc,
                    val_loss=val_loss,
                )
                if backup_path:
                    print(f"✅ Backup created: {backup_path}")

            # Clear memory between epochs
            gc.collect()

    except KeyboardInterrupt:
        print(
            f"\n⚠️ Training interrupted by user at epoch {epoch if 'epoch' in locals() else 'unknown'}"
        )
        # Don't re-raise, just save emergency checkpoint

    except Exception as e:
        print(f"\n❌ Training interrupted with error: {e}")
        print("Saving emergency checkpoint...")

    finally:
        # Always save emergency checkpoint
        emergency_path = run_folder / f"emergency_checkpoint_{unique_run_name}.pth"
        try:
            emergency_data = {
                "epoch": epoch if "epoch" in locals() else 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "training_parameters": training_parameters,
            }
            success = safe_save_checkpoint(emergency_data, emergency_path)
            if success:
                print(f"✅ Emergency checkpoint saved to: {emergency_path}")
            else:
                print("❌ Failed to save emergency checkpoint")
        except Exception as save_error:
            print(f"❌ Failed to save emergency checkpoint: {save_error}")

        if use_wandb:
            cleanup_wandb_run()

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    training_parameters["final_metrics"] = {
        "best_val_accuracy": best_val_acc,
        "final_val_f1": history["val_f1"][-1] if history["val_f1"] else 0,
        "final_train_accuracy": history["train_acc"][-1] if history["train_acc"] else 0,
    }

    save_training_parameters(run_folder, training_parameters)

    # Clear memory before final backup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_backup_path = backup_manager.create_backup(
        model=model,
        optimizer=optimizer,
        epoch=num_epochs - 1 if "epoch" not in locals() else epoch,
        history=history,
        val_acc=val_acc if "val_acc" in locals() else 0,
        val_loss=val_loss if "val_loss" in locals() else 0,
        is_final=True,
    )
    if final_backup_path:
        print(f"✅ Final backup created: {final_backup_path}")

    if use_wandb:
        cleanup_wandb_run()

    return model, history, run_folder


def load_lora_model(
    base_model,
    lora_adapter_path,
    device: torch.device = DEVICE,
    unfreeze_classifier: bool = False,
):
    """
    Load LoRA adapters onto a base model.

    Args:
        base_model: The base model to load LoRA onto
        lora_adapter_path: Path to saved LoRA adapters
        device: Device to load the model to
        unfreeze_classifier: Whether to unfreeze classifier head (for continued training)
    """
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    if unfreeze_classifier:
        # Unfreeze classifier head if needed
        for name, param in model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True

    model.to(device)
    model.eval()

    print(f"Loaded LoRA model from: {lora_adapter_path}")
    model.print_trainable_parameters()

    return model


def load_lora_model_for_inference(
    run_folder, device: torch.device = DEVICE, load_method: str = "auto"
):
    """
    Load a trained LoRA model for inference.

    Args:
        run_folder: Folder containing the saved model
        device: Device to load the model to
        load_method: "auto", "merged", or "lora" to specify loading strategy
    """
    print(f"Loading model from: {run_folder.name}")

    lora_adapter_path = run_folder / "lora_adapter"
    full_checkpoint_path = run_folder / "best_lora_model.pth"
    merged_model_path = run_folder / "merged_model"

    # Try merged model first if requested
    if load_method in ["auto", "merged"] and merged_model_path.exists():
        print("Loading merged model...")
        from transformers import ViTForImageClassification

        try:
            model = ViTForImageClassification.from_pretrained(str(merged_model_path))
            model.to(device)
            model.eval()
            print("Loaded merged model successfully")
            return model
        except Exception as e:
            print(f"Failed to load merged model: {e}")
            if load_method == "merged":
                raise

    # Try LoRA model with classifier weights
    if load_method in ["auto", "lora"] and lora_adapter_path.exists():
        print("Loading LoRA model with classifier weights...")

        # Load checkpoint for metadata and classifier weights
        base_model_name = "google/vit-base-patch16-224-in21k"
        num_labels = 7
        classifier_state = None

        if full_checkpoint_path.exists():
            checkpoint = torch.load(
                full_checkpoint_path, map_location="cpu", weights_only=False
            )
            base_model_name = checkpoint.get(
                "base_model_name", "google/vit-base-patch16-224-in21k"
            )
            num_labels = checkpoint.get("num_labels", 7)
            classifier_state = checkpoint.get("classifier_state", None)
        else:
            # Try to find any .pth file
            pth_files = list(run_folder.glob("*.pth"))
            if pth_files:
                checkpoint = torch.load(
                    pth_files[0], map_location="cpu", weights_only=False
                )
                base_model_name = checkpoint.get(
                    "base_model_name", "google/vit-base-patch16-224-in21k"
                )
                num_labels = checkpoint.get("num_labels", 7)
                classifier_state = checkpoint.get("classifier_state", None)

        # Load base model
        from transformers import ViTForImageClassification

        base_model = ViTForImageClassification.from_pretrained(
            base_model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )

        # Load LoRA adapters and merge
        model = load_lora_model(base_model, lora_adapter_path, device="cpu")
        model = model.merge_and_unload()

        # Apply classifier weights if available
        if classifier_state:
            # Load classifier weights
            for name, weight in classifier_state.items():
                # Navigate to the parameter
                parts = name.split(".")
                param_obj = model
                for part in parts:
                    if hasattr(param_obj, part):
                        param_obj = getattr(param_obj, part)
                    else:
                        break

                # If we found the parameter, update it
                if hasattr(param_obj, "data"):
                    param_obj.data = weight.clone()
                    print(f"Loaded classifier weight: {name}")

        # Alternatively, try to load from model_state_dict
        elif "checkpoint" in locals() and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

            # Extract classifier weights
            classifier_weights = {}
            for key, value in state_dict.items():
                if "classifier" in key or "head" in key:
                    classifier_weights[key] = value

            # Apply classifier weights
            model_state_dict = model.state_dict()
            for key, weight in classifier_weights.items():
                if key in model_state_dict:
                    model_state_dict[key] = weight.clone()
                    print(f"Loaded classifier weight from checkpoint: {key}")

            model.load_state_dict(model_state_dict, strict=False)

        model.to(device)
        model.eval()
        print("Loaded LoRA model successfully")
        return model

    raise FileNotFoundError(f"No valid model files found in {run_folder}")


def merge_and_save_lora_model(lora_model, save_path, save_pretrained: bool = True):
    merged_model = lora_model.merge_and_unload()

    if save_pretrained:
        merged_model.save_pretrained(save_path)
        print(f"Merged model saved to: {save_path}")
    else:
        torch.save(merged_model.state_dict(), save_path / "merged_model.pth")
        print(f"Merged model weights saved to: {save_path / 'merged_model.pth'}")

    return merged_model


def evaluate_lora_model(
    model, test_dataset, batch_size: int = 32, device: torch.device = DEVICE
):
    model.eval()
    model.to(device)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    all_preds = []
    all_labels = []
    all_probs = []

    print("Evaluating model...")
    progress_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
    )

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support_per_class": support_per_class.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return results
