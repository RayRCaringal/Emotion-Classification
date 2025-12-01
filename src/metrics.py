"""
Metrics calculation and reporting utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    average: str = "weighted",
) -> dict:
    accuracy = accuracy_score(labels, predictions)

    if average is not None:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        return {
            "accuracy": accuracy,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support_per_class": support,
        }


def calculate_comprehensive_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict:
    """
    Calculate overall and per-class metrics.
    """

    overall = calculate_metrics(labels, predictions, average="weighted")
    per_class = calculate_metrics(labels, predictions, average=None)

    cm = confusion_matrix(labels, predictions)

    # Classification report as dict
    report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )

    return {
        "accuracy": overall["accuracy"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "precision_per_class": per_class["precision_per_class"],
        "recall_per_class": per_class["recall_per_class"],
        "f1_per_class": per_class["f1_per_class"],
        "support_per_class": per_class["support_per_class"],
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": predictions,
        "labels": labels,
    }
