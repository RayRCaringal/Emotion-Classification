"""
Evaluation functions for emotion classification models with Weights & Biases.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from tqdm.auto import tqdm
import wandb
from typing import Dict

from .config import (
    DEVICE,
    EMOTION_LABELS,
    DEFAULT_BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)


def evaluate_model(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = DEVICE,
    log_to_wandb: bool = True,
    run_name: str = "evaluation"
) -> Dict[str, np.ndarray]:
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
    model.to(device)
    model.eval()
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
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
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    # Log to W&B
    if log_to_wandb:
        # Log overall metrics
        wandb.log({
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
        })
        
        # Log confusion matrix
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=EMOTION_LABELS
            )
        })
        
        # Log per-class metrics as a table
        class_metrics = []
        for i, emotion in enumerate(EMOTION_LABELS):
            class_metrics.append([
                emotion,
                precision_per_class[i],
                recall_per_class[i],
                f1_per_class[i],
                support_per_class[i]
            ])
        
        metrics_table = wandb.Table(
            columns=["Emotion", "Precision", "Recall", "F1-Score", "Support"],
            data=class_metrics
        )
        
        wandb.log({"per_class_metrics": metrics_table})
        
        # Log classification report
        report = classification_report(
            all_labels, all_preds, target_names=EMOTION_LABELS, output_dict=True
        )
        wandb.log({"classification_report": report})
    
    print(f"\n{'='*70}")
    print("Evaluation Results:")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'='*70}\n")
    
    return metrics

def print_classification_report(metrics: Dict[str, np.ndarray]) -> None:
    """
    Print a detailed classification report.
    
    Parameters
    ----------
    metrics : Dict[str, np.ndarray]
        Dictionary from evaluate_model() containing:
        - labels: True labels
        - predictions: Predicted labels
    """
    print(f"\n{'='*70}")
    print("Per-Class Performance:")
    print(f"{'='*70}")
    
    report = classification_report(
        metrics['labels'],
        metrics['predictions'],
        target_names=EMOTION_LABELS,
        digits=4
    )
    print(report)