"""
Evaluation functions for emotion classification models with Weights & Biases.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint_utils import load_model_from_checkpoint
from .config import (
    DEFAULT_BATCH_SIZE,
    DEVICE,
    EMOTION_LABELS,
    NUM_WORKERS,
    PIN_MEMORY,
)
from .metadata import (
    find_all_training_runs,
    find_latest_run_for_experiment,
    load_training_history,
    load_training_parameters,
)
from .metrics import calculate_comprehensive_metrics
from .visualization import create_performance_plot, print_experiment_summary
from .wandb_utils import get_wandb_mode


def config_from_name(run_name: str) -> Dict[str, Any]:

    run_name_lower = run_name.lower()
    
    transform_key = "unknown"
    if "none" in run_name_lower:
        transform_key = "none"
    elif "light" in run_name_lower:
        transform_key = "light"
    elif "medium" in run_name_lower:
        transform_key = "medium"
    elif "heavy" in run_name_lower:
        transform_key = "heavy"
    
    return {
        "transform_key": transform_key,
    }


def load_model_history(
    model_path: Path,
    run_folder: Optional[Path] = None
) -> tuple[torch.nn.Module, Dict, Dict]:
    """ 
    Returns
    -------
    tuple:
        (model, history, training_params)
    """
    model = load_model_from_checkpoint(model_path)
    
    if run_folder is None:
        run_folder = model_path.parent
    
    # Load training history and parameters
    history = load_training_history(run_folder)
    training_params = load_training_parameters(run_folder)
    
    return model, history, training_params


def create_eval_summary(
    run_info: Dict[str, Any],
    test_metrics: Dict[str, Any],
    history: Dict[str, Any],
    training_params: Dict[str, Any]
) -> Dict[str, Any]:

    config = config_from_name(run_info.get("name", run_info.get("run_name", "")))
    
    # Get training metrics from history
    best_val_acc = max(history.get("val_acc", [0]))
    best_val_loss = min(history.get("val_loss", [10]))
    final_train_acc = history.get("train_acc", [0])[-1] if history.get("train_acc") else 0
    
    # Extract training parameters with defaults
    learning_rate = training_params.get("learning_rate", 2e-5)
    epochs = training_params.get("num_epochs", 10)
    batch_size = training_params.get("batch_size", 32)
    
    # Create summary
    summary = {
        "experiment": run_info.get("experiment", run_info.get("name", "")),
        "run_name": run_info.get("run_name", run_info.get("name", "")),
        "transform": config["transform_key"],
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "test_accuracy": test_metrics.get("accuracy", 0),
        "test_precision": test_metrics.get("precision", 0),
        "test_recall": test_metrics.get("recall", 0),
        "test_f1": test_metrics.get("f1", 0),
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_train_accuracy": final_train_acc,
        "run_folder": str(run_info.get("folder", "")),
    }
    
    return summary


def safe_evaluate_model(
    run_info: Dict[str, Any],
    test_dataset: torch.utils.data.Dataset,
    log_to_wandb: bool = False
) -> Optional[Dict[str, Any]]:

    try:

        model_path = run_info.get("best_model_path")
        if model_path is None:
            run_folder = run_info.get("folder")
            if run_folder:
                model_path = Path(run_folder) / f"best_{Path(run_folder).name}.pth"
        
        if model_path is None or not Path(model_path).exists():
            print(f"Model checkpoint not found for {run_info.get('name', 'unknown')}")
            return None
        
        # Load model, history, and parameters
        model, history, training_params = load_model_history(
            model_path=model_path,
            run_folder=run_info.get("folder")
        )
        
        # Evaluate on test set
        test_metrics = evaluate_model(
            model=model,
            test_dataset=test_dataset,
            log_to_wandb=log_to_wandb
        )
        
        summary = create_eval_summary(
            run_info=run_info,
            test_metrics=test_metrics,
            history=history,
            training_params=training_params
        )
        
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        return summary
        
    except Exception as e:
        print(f"Failed to evaluate {run_info.get('name', 'unknown')}: {e}")
        return None


def run_eval(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = DEVICE,
) -> dict:
    model.to(device)
    model.eval()

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

    progress_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(pixel_values=images)
            logits = outputs.logits

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(all_labels, all_preds)
    metrics["probabilities"] = all_probs
    metrics["predictions"] = all_preds
    metrics["labels"] = all_labels

    print(f"\nAccuracy: {metrics['accuracy']:.4f} | F1 Score: {metrics['f1']:.4f}")
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = DEVICE,
    log_to_wandb: bool = True,
    run_name: str = "evaluation",
) -> dict:

    print("Evaluating model...")
    
    # Remove WandB hooks before evaluation to prevent interference
    if hasattr(model, "_wandb_hooks"):
        for hook in getattr(model, "_wandb_hooks", []):
            hook.remove()
        delattr(model, "_wandb_hooks")

    metrics = run_eval(
        model=model, test_dataset=test_dataset, batch_size=batch_size, device=device
    )

    # Log to W&B
    if log_to_wandb and get_wandb_mode() != "disabled":
        import wandb

        wandb.log(
            {
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
            }
        )

        # Log per-class metrics as a table
        class_metrics = []
        for i, emotion in enumerate(EMOTION_LABELS):
            class_metrics.append(
                [
                    emotion,
                    metrics["precision_per_class"][i],
                    metrics["recall_per_class"][i],
                    metrics["f1_per_class"][i],
                    metrics["support_per_class"][i],
                ]
            )

        metrics_table = wandb.Table(
            columns=["Emotion", "Precision", "Recall", "F1-Score", "Support"],
            data=class_metrics,
        )

        wandb.log({"per_class_metrics": metrics_table})

        # Log classification report
        wandb.log({"classification_report": metrics["classification_report"]})

    print(f"\n{'=' * 70}")
    print("Evaluation Results")
    print(f"{'=' * 70}")
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
    print(f"{'=' * 70}")

    return metrics


def evaluate_all_saved_models(
    test_dataset: torch.utils.data.Dataset, checkpoints_dir: Path = None
) -> list[dict]:
    """
    Evaluate all saved models independently
    """
    print(" Evaluating all saved models independently...")

    runs = find_all_training_runs(checkpoints_dir)
    summary_data = []

    for run_info in runs:
        print(f"\n Evaluating: {run_info['name']}")

        try:
            # Load model
            model = load_model_from_checkpoint(run_info["best_model_path"])

            # Load training history and parameters
            history = load_training_history(run_info["folder"])
            training_params = load_training_parameters(run_info["folder"])

            # Evaluate on test set
            test_metrics = evaluate_model(
                model=model, test_dataset=test_dataset, log_to_wandb=False
            )

            summary = create_eval_summary(
                run_info=run_info,
                test_metrics=test_metrics,
                history=history,
                training_params=training_params
            )

            summary_data.append(summary)
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"Failed to evaluate {run_info['name']}: {e}")
            continue

    # Sort by test accuracy (best first)
    summary_data.sort(key=lambda x: x["test_accuracy"], reverse=True)

    # Create visualization
    if len(summary_data) > 1:
        create_performance_plot(summary_data)

    # Print summary
    print_experiment_summary(summary_data)

    return summary_data


def evaluate_specific_models(
    experiment_names: list[str],
    test_dataset: torch.utils.data.Dataset,
    checkpoints_dir: Path = None,
    use_latest: bool = True,
) -> list[dict]:
    """
    Evaluate specific models by experiment name.
    """
    print(f" Evaluating {len(experiment_names)} specific models...")

    summary_data = []

    for exp_name in experiment_names:
        print(f"\n Targeting: {exp_name}")

        try:
            if use_latest:
                run_folder = find_latest_run_for_experiment(exp_name, checkpoints_dir)
            else:
                # Use the base name (original run)
                if checkpoints_dir is None:
                    from .config import CHECKPOINTS_DIR
                    checkpoints_dir = CHECKPOINTS_DIR
                run_folder = checkpoints_dir / exp_name

            # Check if run folder exists
            if not run_folder.exists():
                print(f" Run folder not found: {run_folder}")
                continue

            # Look for the best model checkpoint
            best_model_path = run_folder / f"best_{run_folder.name}.pth"

            if not best_model_path.exists():
                print(f" Model checkpoint not found: {best_model_path}")
                continue

            # Load model, history, and parameters using helper
            model, history, training_params = load_model_history(
                model_path=best_model_path,
                run_folder=run_folder
            )

            # Evaluate on test set
            test_metrics = evaluate_model(
                model=model, test_dataset=test_dataset, log_to_wandb=False
            )

            run_info = {
                "experiment": exp_name,
                "name": exp_name,
                "run_name": run_folder.name,
                "folder": run_folder,
                "best_model_path": best_model_path
            }

            summary = create_eval_summary(
                run_info=run_info,
                test_metrics=test_metrics,
                history=history,
                training_params=training_params
            )

            summary_data.append(summary)
            print(f"    Test Accuracy: {test_metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"    Failed to evaluate {exp_name}: {e}")
            continue

    # Sort by test accuracy (best first)
    summary_data.sort(key=lambda x: x["test_accuracy"], reverse=True)

    # Create visualization
    if len(summary_data) > 1:
        create_performance_plot(summary_data)

    # Print summary
    print_experiment_summary(summary_data)

    return summary_data


def evaluate_from_experiment_configs(
    experiment_configs: list[dict],
    test_dataset: torch.utils.data.Dataset,
    checkpoints_dir: Path = None,
) -> list[dict]:
    experiment_names = [config["name"] for config in experiment_configs]

    print(f"Evaluating {len(experiment_names)} models from experiment configs...")
    for config in experiment_configs:
        print(f"  - {config['name']}: {config['transform_key']} transforms")

    return evaluate_specific_models(
        experiment_names, test_dataset, checkpoints_dir, use_latest=True
    )