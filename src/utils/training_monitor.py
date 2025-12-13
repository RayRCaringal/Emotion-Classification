"""
Training Monitor that serves as a wrapper for handling backups
"""

from pathlib import Path
from typing import Any, Optional, Dict, Callable
from dataclasses import dataclass
import json

import torch
import torch.nn as nn

from src.checkpoint_utils import (
    safe_save_checkpoint, 
    save_model_checkpoint,
    load_checkpoint_for_training
)

@dataclass
class MonitorConfig:
    """Configuration for the training monitor."""
    run_folder: Path
    model_name: str = "model"
    max_backups: int = 3
    backup_interval: int = 5
    save_best: bool = True
    save_final: bool = True
    save_emergency: bool = True
    emergency_run_name: str = "emergency_checkpoint"


class TrainingMonitor:
    """
    Lightweight coordinator for training monitoring.
    Uses existing utilities for actual checkpoint operations.
    """
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.run_folder = config.run_folder
        
        # Create necessary directories
        self.run_folder.mkdir(parents=True, exist_ok=True)
        self.backups_dir = self.run_folder / "backups"
        self.backups_dir.mkdir(exist_ok=True)
        
        # State tracking
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # File paths
        self.best_model_path = self.run_folder / f"best_{self.config.model_name}.pth"
        self.final_model_path = self.run_folder / f"final_{self.config.model_name}.pth"
        self.emergency_path = self.run_folder / f"emergency_{self.config.emergency_run_name}.pth"
        
        print(f"📊 Training Monitor initialized for: {self.config.model_name}")
        print(f"   Run folder: {self.run_folder}")
    
    def register_epoch_update(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        val_acc: float = 0.0,
        val_loss: float = 0.0,
        val_f1: float = 0.0,
        history: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """
        Register an epoch update - call this at the end of each epoch.
        """
        self.current_epoch = epoch
        
        # Save best model if improved
        if self.config.save_best and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self._save_best_checkpoint(
                model, optimizer, epoch, 
                val_acc, val_loss, val_f1, 
                history, lr_scheduler
            )
        
        # Create periodic backup
        if epoch % self.config.backup_interval == 0 or epoch == 0:
            self._create_backup(
                model, optimizer, epoch,
                val_acc, val_loss, val_f1,
                history, lr_scheduler
            )
    
    def _save_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_acc: float,
        val_loss: float,
        val_f1: float,
        history: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """Save the best model checkpoint."""

        if hasattr(model, 'get_checkpoint_metadata'):
            training_parameters = model.get_checkpoint_metadata()
        else:
            training_parameters = {}
        
        success = save_model_checkpoint(
            model=model,
            save_path=self.best_model_path,
            optimizer=optimizer,
            epoch=epoch,
            val_acc=val_acc,
            val_loss=val_loss,
            val_f1=val_f1,
            history=history,
            additional_data={
                "checkpoint_type": "best",
                "best_epoch": epoch,
                "best_val_acc": val_acc,
                "training_parameters": training_parameters,
            }
        )
        
        if success:
            print(f"✅ Best model saved (epoch {epoch}, acc: {val_acc:.4f})")
    
    def _create_backup(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_acc: float,
        val_loss: float,
        val_f1: float,
        history: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Optional[Path]:
        """Create a backup checkpoint using existing utilities."""
        from datetime import datetime
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_epoch_{epoch:03d}_{timestamp}.pth"
            backup_path = self.backups_dir / backup_name
            
            success = save_model_checkpoint(
                model=model,
                save_path=backup_path,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                val_loss=val_loss,
                val_f1=val_f1,
                history=history,
                additional_data={
                    "checkpoint_type": "backup",
                    "backup_timestamp": timestamp,
                }
            )
            
            if success:
                # Save backup metadata
                backup_info = {
                    "backup_path": str(backup_path),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "timestamp": timestamp,
                    "has_lr_scheduler": lr_scheduler is not None,
                }
                
                info_path = backup_path.with_suffix(".json")
                with open(info_path, "w") as f:
                    json.dump(backup_info, f, indent=2)
                
                # Clean up old backups
                self._cleanup_old_backups()
                
                print(f"📦 Backup created: {backup_path.name}")
                return backup_path
            
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
        
        return None
    
    def save_final_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Dict[str, Any],
        training_parameters: Dict[str, Any],
    ) -> None:
        """Save final checkpoint"""
        success = save_model_checkpoint(
            model=model,
            save_path=self.final_model_path,
            optimizer=optimizer,
            epoch=self.current_epoch,
            val_acc=self.best_val_acc,
            history=history,
            additional_data={
                "checkpoint_type": "final",
                "training_parameters": training_parameters,
                "best_val_acc": self.best_val_acc,
                "best_epoch": self.best_epoch,
            }
        )
        
        if success:
            print(f"✅ Final model saved (epoch {self.current_epoch})")
    
    def save_emergency_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Dict[str, Any],
        training_parameters: Dict[str, Any],
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """Save emergency checkpoint directly."""
        if not self.config.save_emergency:
            return
        
        emergency_data = {
            "epoch": self.current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "training_parameters": training_parameters,
            "checkpoint_type": "emergency",
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
        }
        
        if lr_scheduler is not None:
            emergency_data["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        
        success = safe_save_checkpoint(emergency_data, self.emergency_path)
        
        if success:
            print(f" Emergency checkpoint saved to: {self.emergency_path.name}")
    
    def _cleanup_old_backups(self) -> int:
        """Remove old backups beyond max_backups limit."""
        try:
            backup_files = list(self.backups_dir.glob("backup_*.pth"))
            
            if len(backup_files) <= self.config.max_backups:
                return 0
            
            # Sort by time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            backups_to_delete = backup_files[self.config.max_backups:]
            deleted_count = 0
            
            for backup_path in backups_to_delete:
                backup_path.unlink(missing_ok=True)
                
                info_path = backup_path.with_suffix(".json")
                info_path.unlink(missing_ok=True)
                
                deleted_count += 1
            
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} old backups")
            
            return deleted_count
            
        except Exception as e:
            print(f"⚠️ Error cleaning up backups: {e}")
            return 0
    
    def cleanup_after_success(self) -> None:
        """Clean up after successful training."""
        # Delete all backups
        backup_files = list(self.backups_dir.glob("backup_*.pth"))
        for backup_path in backup_files:
            backup_path.unlink(missing_ok=True)
            info_path = backup_path.with_suffix(".json")
            info_path.unlink(missing_ok=True)
        
        # Remove empty backups directory
        try:
            if self.backups_dir.exists() and not any(self.backups_dir.iterdir()):
                self.backups_dir.rmdir()
                print(f"🧹 Removed empty backups directory")
        except Exception as e:
            print(f"⚠️ Could not remove backups directory: {e}")


def monitor_training(
    training_func: Callable,
    config: MonitorConfig,
    *args,
    **kwargs
) -> Any:
    monitor = TrainingMonitor(config)
    
    kwargs['monitor'] = monitor
    
    try:
        # Run the training function
        result = training_func(*args, **kwargs)
        
        # Extract results (assuming standard format)
        if isinstance(result, tuple) and len(result) >= 3:
            model, history, run_folder = result[:3]

            training_parameters = kwargs.get('training_parameters', {})
            
            # Save final checkpoint
            monitor.save_final_checkpoint(
                model=model,
                optimizer=kwargs.get('optimizer'),
                history=history,
                training_parameters=training_parameters,
            )
            
            monitor.cleanup_after_success()
        
        return result
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        if 'model' in kwargs and 'optimizer' in kwargs:
            monitor.save_emergency_checkpoint(
                model=kwargs['model'],
                optimizer=kwargs['optimizer'],
                history=kwargs.get('history', {}),
                training_parameters=kwargs.get('training_parameters', {}),
                lr_scheduler=kwargs.get('lr_scheduler'),
            )
        
        raise KeyboardInterrupt("Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        # Attempt to save emergency checkpoint
        if 'model' in kwargs and 'optimizer' in kwargs:
            monitor.save_emergency_checkpoint(
                model=kwargs['model'],
                optimizer=kwargs['optimizer'],
                history=kwargs.get('history', {}),
                training_parameters=kwargs.get('training_parameters', {}),
                lr_scheduler=kwargs.get('lr_scheduler'),
            )
        raise