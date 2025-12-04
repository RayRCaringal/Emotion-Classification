"""
LoRA (Low-Rank Adaptation) training functions for Vision Transformer fine-tuning.
"""

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

    Args:
        base_model: The base model to apply LoRA to
        lora_config: LoRA configuration
        unfreeze_classifier: Whether to unfreeze the classifier head for training
    """
    model = get_peft_model(base_model, lora_config)

    if unfreeze_classifier:
        # Unfreeze the classifier head for training
        # Note: Different ViT models might have different classifier attribute names
        classifier_attr = None

        # Try to find the classifier attribute
        possible_names = ["classifier", "classifier_head", "head", "fc"]
        for name in possible_names:
            if hasattr(model.base_model, name):
                classifier_attr = name
                break
            elif hasattr(model.base_model.model, name):
                classifier_attr = "model." + name
                break

        if classifier_attr:
            # Handle nested attribute access
            if "." in classifier_attr:
                parent, child = classifier_attr.split(".")
                parent_obj = getattr(model.base_model, parent)
                classifier = getattr(parent_obj, child)
            else:
                classifier = getattr(model.base_model, classifier_attr)

            for param in classifier.parameters():
                param.requires_grad = True

            print(f"✅ Classifier head '{classifier_attr}' unfrozen for training")
        else:
            # Fallback: try to find classifier in named_parameters
            for name, param in model.named_parameters():
                if "classifier" in name.lower() or "head" in name.lower():
                    param.requires_grad = True
                    print(f"✅ Unfroze parameter: {name}")

    model.print_trainable_parameters()
    return model


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
            print(
                f"⚠️ WARNING: Loss doesn't require gradient at epoch {epoch}, batch {batch_idx}"
            )
            print("This might indicate that no parameters require gradients.")

            # Debug: Check which parameters require grad
            trainable_params = [
                (name, p.requires_grad)
                for name, p in model.named_parameters()
                if p.requires_grad
            ]
            if not trainable_params:
                print("❌ CRITICAL: No parameters require gradients!")
            else:
                print(
                    f"✅ Found {len(trainable_params)} parameters with gradients enabled"
                )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

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
    model: PeftModel,
    run_folder,
    model_name: str = "lora_model",
    save_classifier_separately: bool = True,
):
    """
    Save LoRA model for inference with options to save classifier weights separately.

    Args:
        model: The trained LoRA model
        run_folder: Folder to save the model
        model_name: Base name for the model files
        save_classifier_separately: Whether to save classifier weights separately for reloading
    """
    print("\nSaving model for inference...")

    lora_save_path = run_folder / "lora_adapter"
    best_model_path = run_folder / f"best_{model_name}.pth"

    # Save LoRA adapters
    model.save_pretrained(lora_save_path)
    print(f"LoRA adapters saved to: {lora_save_path}")

    # Save full model checkpoint with classifier weights
    # Extract classifier weights if they exist
    classifier_state = {}
    for name, param in model.named_parameters():
        if "classifier" in name or "head" in name:
            classifier_state[name] = param.data.clone()

    # Get base model info
    base_model_name = (
        model.base_model.config._name_or_path
        if hasattr(model.base_model.config, "_name_or_path")
        else "google/vit-base-patch16-224-in21k"
    )
    num_labels = (
        model.base_model.config.num_labels
        if hasattr(model.base_model.config, "num_labels")
        else 7
    )

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "lora_config": model.peft_config["default"].to_dict()
        if hasattr(model, "peft_config")
        else {},
        "base_model_name": base_model_name,
        "num_labels": num_labels,
    }

    # Add classifier weights if found
    if classifier_state:
        checkpoint["classifier_state"] = classifier_state
        print(f"Saved classifier weights for {len(classifier_state)} parameters")

    torch.save(checkpoint, best_model_path)
    print(f"Full model checkpoint saved to: {best_model_path}")

    # Optionally save merged model
    try:
        merged_model = model.merge_and_unload()
        merged_save_path = run_folder / "merged_model"
        merged_model.save_pretrained(merged_save_path)
        print(f"Merged model saved to: {merged_save_path}")
        return {
            "lora_adapters": lora_save_path,
            "full_checkpoint": best_model_path,
            "merged_model": merged_save_path,
        }
    except Exception as e:
        print(f"⚠️ Could not save merged model: {e}")
        print("Saving only LoRA adapters and checkpoint")
        return {
            "lora_adapters": lora_save_path,
            "full_checkpoint": best_model_path,
            "merged_model": None,
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
):
    from src.wandb_utils import cleanup_wandb_run, init_wandb_run, log_epoch

    run_folder = get_next_folder(f"{model_name}")
    unique_run_name = run_folder.name

    # Count trainable parameters BEFORE moving to device
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    training_parameters = {
        "model_name": model_name,
        "model_type": "LoRA",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
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
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=DEFAULT_WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    # Move model to device AFTER creating loaders
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

    # Debug: Print trainable parameter names
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

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

                save_paths = save_lora_model_for_inference(
                    model=model, run_folder=run_folder, model_name=model_name
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "val_f1": val_f1,
                        "history": history,
                    },
                    best_model_path,
                )
                print(f"Model saved (Val Acc: {val_acc:.4f})")

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
        print(f"\nTraining interrupted with error: {e}")
        print("Saving current state before exit...")

        emergency_path = run_folder / f"emergency_checkpoint_{unique_run_name}.pth"
        torch.save(
            {
                "epoch": epoch if "epoch" in locals() else 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            emergency_path,
        )
        print(f"Emergency checkpoint saved to: {emergency_path}")

        if use_wandb:
            cleanup_wandb_run()

        raise

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    training_parameters["final_metrics"] = {
        "best_val_accuracy": best_val_acc,
        "final_val_f1": history["val_f1"][-1],
        "final_train_accuracy": history["train_acc"][-1],
    }

    save_training_parameters(run_folder, training_parameters)

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
