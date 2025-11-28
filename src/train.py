"""
Training functions for fine tuning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import json
import wandb

from .config import (
    DEVICE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WARMUP_STEPS,
    NUM_WORKERS,
    PIN_MEMORY,
    CHECKPOINTS_DIR
)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int
) -> tuple[float, float, float, float, float, list, list]:
    """
    Train the model for one epoch.

    Returns
    -------
    avg_loss : 
        Average loss for the epoch
    accuracy : 
        Accuracy for the epoch
    precision : 
        Weighted precision score
    recall : 
        Weighted recall score
    f1 : 
        Weighted F1-score
    all_preds : 
        All predictions
    all_labels : 
        All true labels
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward Pass
        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 50 == 0:
            wandb.log({
                'batch_train_loss': loss.item(),
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'epoch': epoch,
                'batch': batch_idx
            })
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int
) -> tuple[float, float, float, float, float, list, list]:
    """
    Validate one epoch.

    Returns
    -------
    avg_loss : float
        Average loss
    accuracy : float
        Accuracy
    precision : float
        Weighted precision
    recall : float
        Weighted recall
    f1 : float
        Weighted F1-score
    all_preds : list
        All predictions
    all_labels : list
        All true labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch}")
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = DEVICE,
    save_path: Path = None,
    model_name: str = "model",
    use_wandb: bool = True,
    wandb_config: dict = None
) -> tuple[torch.nn.Module, dict]:
    """
    Returns
    -------
    model : torch.nn.Module
        Trained model
    history : dict
        Dictionary with training history
    """

    if use_wandb:
        if wandb_config is None:
            wandb_config = {
                "learning_rate": DEFAULT_LEARNING_RATE,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "model_name": model_name,
                "architecture": type(model).__name__,
            }
        
        wandb.init(
            project="emotion-classification",
            name=model_name,
            config=wandb_config
        )

        wandb.watch(model, log="all", log_freq=100)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # Learning Scheduler
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=DEFAULT_WARMUP_STEPS,
        num_training_steps=num_training_steps
    )
    
    model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_acc = 0.0
    
    if save_path is None:
        save_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"
    
    print(f"Training {model_name} for {num_epochs} epochs...")
    print(f"Total training steps: {num_training_steps}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Best model will be saved to: {save_path}")
    if use_wandb:
        print(f"W&B tracking: {wandb.run.url}")
    print("="*70)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc, train_precision, train_recall, train_f1, train_preds , train_labels = train_epoch(
            model, train_loader, optimizer, lr_scheduler, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = validate(
            model, val_loader, device, epoch
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'learning_rate': lr_scheduler.get_last_lr()[0]
            })
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
        
        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'history': history
            }, save_path)
            print(f"✓ New best model saved! (Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f})")
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save Final History
    history_path = CHECKPOINTS_DIR / f"history_{model_name}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Finish W&B run
    if use_wandb:
        wandb.finish()
    
    return model, history


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer = None
) -> tuple[torch.nn.Module, dict]:
    """
    Returns
    -------
    model : torch.nn.Module
        Model with loaded weights
    checkpoint : dict
        Full checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"Val F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
    
    return model, checkpoint