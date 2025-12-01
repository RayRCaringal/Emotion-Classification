"""
Evaluation functions for emotion classification models with Weights & Biases.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import (
    DEFAULT_BATCH_SIZE,
    DEVICE,
    EMOTION_LABELS,
    NUM_WORKERS,
    PIN_MEMORY,
)
from .wandb_utils import get_wandb_mode


def evaluate_model(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = DEVICE,
    log_to_wandb: bool = True,
    run_name: str = "evaluation",
) -> dict[str, np.ndarray]:
    """
    Returns
    -------
    metrics :
        Dictionary:
        - accuracy: Overall accuracy
        - precision: Weighted precision score
        - recall: Weighted recall score
        - f1: Weighted F1-score
        - precision_per_class: Per-class precision scores
        - recall_per_class: Per-class recall scores
        - f1_per_class: Per-class F1-scores
        - support_per_class: Number of samples per class
        - confusion_matrix: Confusion matrix
        - predictions: All predictions
        - labels: All true labels
        - probabilities: All prediction probabilities
    """
    # Remove WandB hooks before evaluation to prevent the error
    if hasattr(model, "_wandb_hooks"):
        for hook in getattr(model, "_wandb_hooks", []):
            hook.remove()
        delattr(model, "_wandb_hooks")

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

    print("Evaluating model...")
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

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "support_per_class": support_per_class,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }

    # Log to W&B
    if log_to_wandb and get_wandb_mode() != "disabled":
        import wandb

        # Log overall metrics
        wandb.log(
            {
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            }
        )

        # Log per-class metrics as a table
        class_metrics = []
        for i, emotion in enumerate(EMOTION_LABELS):
            class_metrics.append(
                [
                    emotion,
                    precision_per_class[i],
                    recall_per_class[i],
                    f1_per_class[i],
                    support_per_class[i],
                ]
            )

        metrics_table = wandb.Table(
            columns=["Emotion", "Precision", "Recall", "F1-Score", "Support"],
            data=class_metrics,
        )

        wandb.log({"per_class_metrics": metrics_table})

        # Log classification report
        report = classification_report(
            all_labels, all_preds, target_names=EMOTION_LABELS, output_dict=True
        )
        wandb.log({"classification_report": report})

    print(f"\n{'=' * 70}")
    print("Evaluation Results:")
    print(f"{'=' * 70}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'=' * 70}\n")

    return metrics


def print_classification_report(metrics: dict[str, np.ndarray]) -> None:
    """
    Print a detailed classification report.

    Parameters
    ----------
    metrics : Dict[str, np.ndarray]
        Dictionary from evaluate_model() containing:
        - labels: True labels
        - predictions: Predicted labels
    """
    print(f"\n{'=' * 70}")
    print("Per-Class Performance:")
    print(f"{'=' * 70}")

    report = classification_report(
        metrics["labels"], metrics["predictions"], target_names=EMOTION_LABELS, digits=4
    )
    print(report)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_name: str = "google/vit-base-patch16-224-in21k",
    num_labels: int = 7,
) -> torch.nn.Module:
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"✅ Loaded model from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")

    return model


def load_training_history(run_folder: Path) -> dict:
    history_path = run_folder / f"history_{run_folder.name}.json"

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        print(f"✅ Loaded history from: {history_path}")
        return history
    else:
        print(f"❌ History file not found: {history_path}")
        return {}


def find_all_training_runs(checkpoints_dir: Path = None) -> list[dict]:
    """
    Find all training runs in the checkpoints directory.

    Parameters
    ----------
    checkpoints_dir : Path, optional
        Checkpoints directory (defaults to CHECKPOINTS_DIR from config)

    Returns
    -------
    list[dict]
        List of run information dictionaries
    """
    if checkpoints_dir is None:
        from .config import CHECKPOINTS_DIR

        checkpoints_dir = CHECKPOINTS_DIR

    runs = []

    for run_folder in checkpoints_dir.iterdir():
        if run_folder.is_dir():
            # Look for the best model checkpoint
            best_model_path = run_folder / f"best_{run_folder.name}.pth"

            if best_model_path.exists():
                runs.append(
                    {
                        "name": run_folder.name,
                        "folder": run_folder,
                        "best_model_path": best_model_path,
                        "history_path": run_folder / f"history_{run_folder.name}.json",
                        "params_path": run_folder / "training_parameters.json",
                    }
                )

    print(f"📁 Found {len(runs)} training runs")
    return runs


def evaluate_all_saved_models(
    test_dataset: torch.utils.data.Dataset, checkpoints_dir: Path = None
) -> list[dict]:
    """
    Evaluate all saved models independently (no need for all_results in memory).

    Parameters
    ----------
    test_dataset : Dataset
        Test dataset for evaluation
    checkpoints_dir : Path, optional
        Checkpoints directory

    Returns
    -------
    list[dict]
        List of summary dictionaries
    """
    print("🚀 Evaluating all saved models independently...")

    # Find all training runs
    runs = find_all_training_runs(checkpoints_dir)

    summary_data = []

    for run_info in runs:
        print(f"\n📊 Evaluating: {run_info['name']}")

        try:
            # Load model
            model = load_model_from_checkpoint(run_info["best_model_path"])

            # Load training history
            history = load_training_history(run_info["folder"])

            # Load training parameters
            training_params = {}
            if run_info["params_path"].exists():
                with open(run_info["params_path"]) as f:
                    training_params = json.load(f)

            # Evaluate on test set
            test_metrics = evaluate_model(
                model=model, test_dataset=test_dataset, log_to_wandb=False
            )

            # Extract config from run name and parameters
            transform_key = "unknown"
            learning_rate = training_params.get("learning_rate", 2e-5)
            epochs = training_params.get("num_epochs", 10)
            batch_size = training_params.get("batch_size", 32)

            # Parse transform from run name
            run_name = run_info["name"]
            if "none" in run_name:
                transform_key = "none"
            elif "light" in run_name:
                transform_key = "light"
            elif "medium" in run_name:
                transform_key = "medium"
            elif "heavy" in run_name:
                transform_key = "heavy"

            # Get training metrics from history
            best_val_acc = max(history.get("val_acc", [0]))
            best_val_loss = min(history.get("val_loss", [10]))
            final_train_acc = (
                history.get("train_acc", [0])[-1] if history.get("train_acc") else 0
            )

            summary = {
                "experiment": run_info["name"],
                "transform": transform_key,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "best_val_accuracy": best_val_acc,
                "best_val_loss": best_val_loss,
                "final_train_accuracy": final_train_acc,
                "run_folder": str(run_info["folder"]),
            }

            summary_data.append(summary)
            print(f"   ✅ Test Accuracy: {test_metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"   ❌ Failed to evaluate {run_info['name']}: {e}")
            continue

    # Sort by test accuracy (best first)
    summary_data.sort(key=lambda x: x["test_accuracy"], reverse=True)

    # Create visualization
    if len(summary_data) > 1:
        create_performance_plot(summary_data)

    # Print summary
    print_experiment_summary(summary_data)

    return summary_data


def create_performance_plot(summary_data: list[dict]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract data for plotting
    experiments = [exp["experiment"] for exp in summary_data]
    test_accuracies = [exp["test_accuracy"] for exp in summary_data]
    val_accuracies = [exp["best_val_accuracy"] for exp in summary_data]
    transforms = [exp["transform"] for exp in summary_data]

    # Colors based on transform type
    transform_colors = {
        "none": "skyblue",
        "light": "lightgreen",
        "medium": "gold",
        "heavy": "lightcoral",
    }
    colors = [transform_colors.get(t, "gray") for t in transforms]

    # Plot 1: Test vs Validation Accuracy
    x_pos = np.arange(len(experiments))
    width = 0.35

    bars1 = ax1.bar(
        x_pos - width / 2,
        test_accuracies,
        width,
        label="Test Accuracy",
        color=colors,
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x_pos + width / 2,
        val_accuracies,
        width,
        label="Val Accuracy",
        color=colors,
        alpha=0.6,
    )

    ax1.set_xlabel("Experiments")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Test vs Validation Accuracy by Experiment")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(experiments, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Transform comparison
    transform_accuracies = {}
    for exp in summary_data:
        transform = exp["transform"]
        if transform not in transform_accuracies:
            transform_accuracies[transform] = []
        transform_accuracies[transform].append(exp["test_accuracy"])

    transform_means = {t: np.mean(accs) for t, accs in transform_accuracies.items()}
    transform_stds = {t: np.std(accs) for t, accs in transform_accuracies.items()}

    transform_names = list(transform_means.keys())
    transform_vals = [transform_means[t] for t in transform_names]
    transform_errs = [transform_stds[t] for t in transform_names]
    transform_colors_plot = [transform_colors.get(t, "gray") for t in transform_names]

    bars = ax2.bar(
        transform_names,
        transform_vals,
        yerr=transform_errs,
        capsize=5,
        color=transform_colors_plot,
        alpha=0.8,
    )
    ax2.set_xlabel("Transform Type")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Performance by Transform Type")
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("experiment_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_experiment_summary(summary_data: list[dict]):
    """
    Print a beautifully formatted summary of all experiments.

    Parameters
    ----------
    summary_data : list[dict]
        List of experiment summaries from summarize_experiments()
    """
    print("\n" + "=" * 120)
    print("🏆 EXPERIMENT SUMMARY REPORT")
    print("=" * 120)

    # Header
    header = f"{'Experiment':<20} {'Transform':<10} {'LR':<12} {'Test Acc':<10} {'Test F1':<10} {'Val Acc':<10} {'Train Acc':<10}"
    print(header)
    print("-" * len(header))

    # Data rows
    for exp in summary_data:
        row = (
            f"{exp['experiment']:<20} "
            f"{exp['transform']:<10} "
            f"{exp['learning_rate']:<12.2e} "
            f"{exp['test_accuracy']:<10.4f} "
            f"{exp['test_f1']:<10.4f} "
            f"{exp['best_val_accuracy']:<10.4f} "
            f"{exp['final_train_accuracy']:<10.4f}"
        )
        print(row)

    # Key insights
    best_exp = summary_data[0]
    worst_exp = summary_data[-1]

    print("\n" + "=" * 120)
    print("📈 KEY INSIGHTS")
    print("=" * 120)

    print(f"🏅 BEST PERFORMER: {best_exp['experiment']}")
    print(f"   Test Accuracy: {best_exp['test_accuracy']:.4f}")
    print(f"   Transform: {best_exp['transform']}")
    print(f"   Learning Rate: {best_exp['learning_rate']:.2e}")

    print(f"\n📉 WORST PERFORMER: {worst_exp['experiment']}")
    print(f"   Test Accuracy: {worst_exp['test_accuracy']:.4f}")
    print(f"   Transform: {worst_exp['transform']}")

    # Transform analysis
    print("\n🔄 TRANSFORM ANALYSIS")
    print("-" * 50)
    transform_stats = {}
    for exp in summary_data:
        transform = exp["transform"]
        if transform not in transform_stats:
            transform_stats[transform] = []
        transform_stats[transform].append(exp["test_accuracy"])

    for transform, accuracies in transform_stats.items():
        avg_acc = np.mean(accuracies)
        best_acc = max(accuracies)
        print(f"   {transform:8}: Avg = {avg_acc:.4f} (Best = {best_acc:.4f})")

    all_accuracies = [exp["test_accuracy"] for exp in summary_data]
    print(
        f"\n📏 PERFORMANCE RANGE: {min(all_accuracies):.4f} - {max(all_accuracies):.4f}"
    )
    print(
        f"📊 AVERAGE ACCURACY:  {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}"
    )


def find_latest_run_for_experiment(
    experiment_name: str, checkpoints_dir: Path = None
) -> Path:
    """
    Find the latest run folder for a given experiment name.

    Parameters
    ----------
    experiment_name : str
        Base experiment name (e.g., "baseline_none")
    checkpoints_dir : Path, optional
        Checkpoints directory

    Returns
    -------
    Path
        Path to the latest run folder
    """
    if checkpoints_dir is None:
        from .config import CHECKPOINTS_DIR

        checkpoints_dir = CHECKPOINTS_DIR

    # Pattern to match experiment_name followed by optional numbers
    import re

    pattern = re.compile(rf"{re.escape(experiment_name)}(\d*)$")

    matching_folders = []

    for folder in checkpoints_dir.iterdir():
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                # Extract the number if it exists
                number_str = match.group(1)
                if number_str:
                    folder_number = int(number_str)
                else:
                    folder_number = 0  # No number means it's the original

                matching_folders.append((folder, folder_number))

    if not matching_folders:
        raise FileNotFoundError(f"No runs found for experiment: {experiment_name}")

    # Sort by folder number (highest first)
    matching_folders.sort(key=lambda x: x[1], reverse=True)

    latest_folder = matching_folders[0][0]
    print(f"📁 Latest run for '{experiment_name}': {latest_folder.name}")

    return latest_folder


def evaluate_specific_models(
    experiment_names: list[str],
    test_dataset: torch.utils.data.Dataset,
    checkpoints_dir: Path = None,
    use_latest: bool = True,
) -> list[dict]:
    """
    Evaluate specific models by experiment name.

    Parameters
    ----------
    experiment_names : list[str]
        List of experiment names to evaluate
    test_dataset : Dataset
        Test dataset for evaluation
    checkpoints_dir : Path, optional
        Checkpoints directory
    use_latest : bool
        Whether to use the latest run for each experiment

    Returns
    -------
    list[dict]
        List of summary dictionaries
    """
    print(f"🚀 Evaluating {len(experiment_names)} specific models...")

    summary_data = []

    for exp_name in experiment_names:
        print(f"\n🎯 Targeting: {exp_name}")

        try:
            if use_latest:
                # Find the latest run for this experiment
                run_folder = find_latest_run_for_experiment(exp_name, checkpoints_dir)
            else:
                # Use the base name (original run)
                if checkpoints_dir is None:
                    from .config import CHECKPOINTS_DIR

                    checkpoints_dir = CHECKPOINTS_DIR
                run_folder = checkpoints_dir / exp_name

            # Check if run folder exists
            if not run_folder.exists():
                print(f"❌ Run folder not found: {run_folder}")
                continue

            # Look for the best model checkpoint
            best_model_path = run_folder / f"best_{run_folder.name}.pth"

            if not best_model_path.exists():
                print(f"❌ Model checkpoint not found: {best_model_path}")
                continue

            # Load model
            model = load_model_from_checkpoint(best_model_path)

            # Load training history
            history = load_training_history(run_folder)

            # Load training parameters
            training_params = {}
            params_path = run_folder / "training_parameters.json"
            if params_path.exists():
                with open(params_path) as f:
                    training_params = json.load(f)

            # Evaluate on test set
            test_metrics = evaluate_model(
                model=model, test_dataset=test_dataset, log_to_wandb=False
            )

            # Extract config from run name and parameters
            transform_key = "unknown"
            learning_rate = training_params.get("learning_rate", 2e-5)
            epochs = training_params.get("num_epochs", 10)
            batch_size = training_params.get("batch_size", 32)

            # Parse transform from run name
            run_name = run_folder.name
            if "none" in run_name.lower():
                transform_key = "none"
            elif "light" in run_name.lower():
                transform_key = "light"
            elif "medium" in run_name.lower():
                transform_key = "medium"
            elif "heavy" in run_name.lower():
                transform_key = "heavy"

            # Get training metrics from history
            best_val_acc = max(history.get("val_acc", [0]))
            best_val_loss = min(history.get("val_loss", [10]))
            final_train_acc = (
                history.get("train_acc", [0])[-1] if history.get("train_acc") else 0
            )

            summary = {
                "experiment": exp_name,
                "run_name": run_folder.name,
                "transform": transform_key,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "best_val_accuracy": best_val_acc,
                "best_val_loss": best_val_loss,
                "final_train_accuracy": final_train_acc,
                "run_folder": str(run_folder),
            }

            summary_data.append(summary)
            print(f"   ✅ Test Accuracy: {test_metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"   ❌ Failed to evaluate {exp_name}: {e}")
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
    """
    Evaluate models based on experiment configurations.

    Parameters
    ----------
    experiment_configs : list[dict]
        List of experiment configurations
    test_dataset : Dataset
        Test dataset for evaluation
    checkpoints_dir : Path, optional
        Checkpoints directory

    Returns
    -------
    list[dict]
        List of summary dictionaries
    """
    experiment_names = [config["name"] for config in experiment_configs]

    print(f"🔍 Evaluating {len(experiment_names)} models from experiment configs...")
    for config in experiment_configs:
        print(f"   - {config['name']}: {config['transform_key']} transforms")

    return evaluate_specific_models(
        experiment_names, test_dataset, checkpoints_dir, use_latest=True
    )


def evaluate_manual_folders(
    folder_names: list[str],
    test_dataset: torch.utils.data.Dataset,
    checkpoints_dir: Path = None,
) -> list[dict]:
    """
    Evaluate specific run folders by name.

    Parameters
    ----------
    folder_names : list[str]
        List of exact folder names to evaluate
    test_dataset : Dataset
        Test dataset for evaluation
    checkpoints_dir : Path, optional
        Checkpoints directory

    Returns
    -------
    list[dict]
        List of summary dictionaries
    """
    print(f"📂 Evaluating {len(folder_names)} specific run folders...")

    summary_data = []

    for folder_name in folder_names:
        print(f"\n📁 Evaluating folder: {folder_name}")

        try:
            if checkpoints_dir is None:
                from .config import CHECKPOINTS_DIR

                checkpoints_dir = CHECKPOINTS_DIR

            run_folder = checkpoints_dir / folder_name

            if not run_folder.exists():
                print(f"❌ Folder not found: {run_folder}")
                continue

            # Look for the best model checkpoint
            best_model_path = run_folder / f"best_{folder_name}.pth"

            if not best_model_path.exists():
                # Try alternative naming pattern
                best_model_path = run_folder / f"best_{run_folder.name}.pth"
                if not best_model_path.exists():
                    print(f"❌ Model checkpoint not found in: {folder_name}")
                    continue

            # Load model
            model = load_model_from_checkpoint(best_model_path)

            # Load training history
            history = load_training_history(run_folder)

            # Evaluate on test set
            test_metrics = evaluate_model(
                model=model, test_dataset=test_dataset, log_to_wandb=False
            )

            # Extract experiment name from folder (remove numbers)
            import re

            exp_name = re.sub(r"\d+$", "", folder_name)

            summary = {
                "experiment": exp_name,
                "run_name": folder_name,
                "transform": "unknown",  # Would need to parse from folder name
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "run_folder": str(run_folder),
            }

            summary_data.append(summary)
            print(f"   ✅ Test Accuracy: {test_metrics['accuracy']:.4f}")

        except Exception as e:
            print(f"   ❌ Failed to evaluate {folder_name}: {e}")
            continue

    # Sort by test accuracy (best first)
    summary_data.sort(key=lambda x: x["test_accuracy"], reverse=True)

    # Print summary
    print_experiment_summary(summary_data)

    return summary_data
