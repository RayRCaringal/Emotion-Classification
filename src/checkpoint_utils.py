"""
Checkpoint and state management utilities.
"""

from pathlib import Path
from typing import Any, Optional, Union, Type
import pickle

import torch
import torch.nn as nn

from src.config import DEVICE
from src.models.base_model import BaseModel


def safe_save_checkpoint(state: dict, save_path: Path, verify: bool = True) -> bool:
    """
    Safely saves a torch checkpoint with error handling and verification.
    
    Parameters
    ----------
    verify : bool
        Whether to verify the file was written successfully

    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, save_path, _use_new_zipfile_serialization=True)
        
        if verify:
            if save_path.exists():
                file_size = save_path.stat().st_size
                print(f"Checkpoint saved: {save_path.name} ({file_size:,} bytes)")
                return True
            else:
                print(f"Failed to save checkpoint: {save_path.name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error saving checkpoint {save_path.name}: {e}")
        return False


def safe_load_checkpoint(
    checkpoint_path: Path,
    device: Union[str, torch.device] = DEVICE,
    weights_only: bool = False
) -> Optional[dict]:
    """
    Safely loads a torch checkpoint with error handling and fallback methods.
    
    Parameters
    ----------
    weights_only : bool
        If True, only load model_state_dict for safety  
    """
    try:
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path.name}")
            return None
        
        file_size = checkpoint_path.stat().st_size
        print(f"Loading checkpoint: {checkpoint_path.name} ({file_size:,} bytes)")
        
        # Try loading with weights_only flag
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=weights_only
        )
        
        if checkpoint is not None:
            print("Checkpoint loaded successfully")
            
            # If weights_only requested, extract just the model state
            if weights_only and "model_state_dict" in checkpoint:
                return {"model_state_dict": checkpoint["model_state_dict"]}
            
            return checkpoint
        else:
            print("Checkpoint loaded but is None")
            return None
            
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path.name}: {e}")
        
        try:
            print("Trying alternative loading method (pickle)...")
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            print("Loaded with pickle")
            return checkpoint
        except Exception as pickle_error:
            print(f"Pickle loading also failed: {pickle_error}")
            return None


def save_model_checkpoint(
    model: BaseModel,
    save_path: Path,
    optimizer: torch.optim.Optimizer,
    epoch: int = 0,
    val_acc: float = 0.0,
    val_loss: float = 0.0,
    val_f1: float = 0.0,
    history: Optional[dict] = None,
    additional_data: Optional[dict] = None,
) -> bool:
   
    training_parameters = model.get_checkpoint_metadata()
    
    if history is None:
        history = {}
    
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
    
    if additional_data:
        checkpoint_data.update(additional_data)
    
    success = safe_save_checkpoint(checkpoint_data, save_path)
    
    if success:
        print(f"Training checkpoint saved (Epoch {epoch}, Val Acc: {val_acc:.4f})")
    
    return success


def load_checkpoint_for_training(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = DEVICE
) -> tuple[nn.Module, Optional[torch.optim.Optimizer], dict]:
    """
    Load a checkpoint and restore training state.
    
    Returns
    -------
    tuple[nn.Module, Optional[torch.optim.Optimizer], dict]
        Tuple of (model, optimizer, checkpoint_dict)
    """
    checkpoint = safe_load_checkpoint(checkpoint_path, device=device, weights_only=False)
    
    if checkpoint is None:
        raise FileNotFoundError(f"Could not load checkpoint from {checkpoint_path}")
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", "N/A")
    val_acc = checkpoint.get("val_acc", "N/A")
    val_f1 = checkpoint.get("val_f1", "N/A")
    
    print(f" Loaded checkpoint from: {checkpoint_path.name}")
    print(f"   Epoch: {epoch}")
    print(f"   Val Acc: {val_acc:.4f}" if isinstance(val_acc, float) else f"   Val Acc: {val_acc}")
    print(f"   Val F1: {val_f1:.4f}" if isinstance(val_f1, float) else f"   Val F1: {val_f1}")
    
    return model, optimizer, checkpoint


def load_model_from_checkpoint(
    model_class: Type[BaseModel],
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> BaseModel:
    """
    Load a model from checkpoint.
    
    This creates a new model instance and loads the checkpoint state into it.
    
    Parameters
    ----------
    strict : bool
        Whether to strictly enforce state_dict keys match
        
    Returns
    -------
    BaseModel
        Loaded model instance
    """
    checkpoint = safe_load_checkpoint(
        checkpoint_path, 
        device=device or "cpu", 
        weights_only=False
    )
    
    if checkpoint is None:
        raise FileNotFoundError(f"Could not load checkpoint from {checkpoint_path}")
    

    training_params = checkpoint.get("training_parameters", {})
    num_labels = training_params.get("num_labels", 7)
    base_model_name = training_params.get("base_model_name", "base_model")
    
    model = model_class(
        num_labels=num_labels,
        model_name=base_model_name,
        device=device,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    print(f" Loaded {model_class.__name__} from checkpoint: {checkpoint_path.name}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Acc: {checkpoint.get('val_acc', 0.0):.4f}")
    print(f"   Val F1: {checkpoint.get('val_f1', 0.0):.4f}")
    
    return model


def load_model_for_inference(
    model_class: Type[BaseModel],
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> BaseModel:

    model = load_model_from_checkpoint(
        model_class=model_class,
        checkpoint_path=checkpoint_path,
        device=device,
        strict=False, 
    )
    
    model.eval()
    
    # Disable gradients for inference
    for param in model.parameters():
        param.requires_grad = False
    
    print(" Model set to inference mode (no gradients)")
    
    return model