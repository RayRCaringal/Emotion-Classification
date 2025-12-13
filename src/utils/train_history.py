"""
Training history management functions
"""

from typing import Dict, Any


def init_training_history() -> Dict[str, list]:
    return {
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


def update_history(
    history: Dict[str, list],
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
) -> None:

    # Update training metrics
    history["train_loss"].append(train_metrics.get("loss", 0.0))
    history["train_acc"].append(train_metrics.get("accuracy", 0.0))
    history["train_precision"].append(train_metrics.get("precision", 0.0))
    history["train_recall"].append(train_metrics.get("recall", 0.0))
    history["train_f1"].append(train_metrics.get("f1", 0.0))
    
    # Update validation metrics
    history["val_loss"].append(val_metrics.get("loss", 0.0))
    history["val_acc"].append(val_metrics.get("accuracy", 0.0))
    history["val_precision"].append(val_metrics.get("precision", 0.0))
    history["val_recall"].append(val_metrics.get("recall", 0.0))
    history["val_f1"].append(val_metrics.get("f1", 0.0))


