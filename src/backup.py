"""
Creates Backup checkpoints in case of Training Failure.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict

import torch


class BackupManager:
    """
    Maintains a rotating set of backups (default: 3 most recent).
    """

    def __init__(
        self, run_folder: Path, max_backups: int = 3, backup_interval: int = 5
    ):
        """
        Parameters
        ----------
        max_backups :
            Maximum number of backups to keep
        backup_interval :
            Create backup every N epochs
        """
        self.run_folder = run_folder
        self.max_backups = max_backups
        self.backup_interval = backup_interval
        self.backups_dir = run_folder / "backups"
        self.backups_dir.mkdir(exist_ok=True)

    def should_backup(self, epoch: int) -> bool:
        return (epoch > 0 and epoch % self.backup_interval == 0) or epoch == 0

    def create_backup(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        history: dict[str, Any],
        val_acc: float,
        val_loss: float,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        is_final: bool = False,
    ) -> Optional[Path]:
      
        try:
            backup_type = "final" if is_final else f"epoch_{epoch:03d}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{backup_type}_{timestamp}.pth"
            backup_path = self.backups_dir / backup_name

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "timestamp": datetime.now().isoformat(),
                "backup_type": backup_type,
                "is_final": is_final,
            }
            
            if lr_scheduler is not None:
                checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
                checkpoint["lr_scheduler_type"] = type(lr_scheduler).__name__
            
            if additional_data:
                checkpoint["additional_data"] = additional_data

            torch.save(checkpoint, backup_path)

            # Create JSON metadata file
            backup_info = {
                "backup_path": str(backup_path),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "timestamp": checkpoint["timestamp"],
                "backup_type": backup_type,
                "is_final": is_final,
                "has_lr_scheduler": lr_scheduler is not None,
                "has_additional_data": additional_data is not None,
                "lr_scheduler_type": type(lr_scheduler).__name__ if lr_scheduler else None,
            }

            info_path = backup_path.with_suffix(".json")
            with open(info_path, "w") as f:
                json.dump(backup_info, f, indent=2)

            # Clean up old backups
            self.cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None

    def cleanup_old_backups(self) -> int:
        """
        Remove old backups beyond the maximum allowed.
        Keeps the most recent backups.
        """
        try:
            backup_files = list(self.backups_dir.glob("backup_*.pth"))

            if len(backup_files) <= self.max_backups:
                return 0

            # Sort by time
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            backups_to_keep = backup_files[: self.max_backups]
            backups_to_delete = backup_files[self.max_backups :]

            deleted_count = 0
            for backup_path in backups_to_delete:
                # Delete both .pth and .json files
                if backup_path.exists():
                    backup_path.unlink(missing_ok=True)

                info_path = backup_path.with_suffix(".json")
                if info_path.exists():
                    info_path.unlink(missing_ok=True)

                print(f"Deleted old backup: {backup_path.name}")
                deleted_count += 1

            return deleted_count

        except Exception as e:
            print(f"Error during backup cleanup: {e}")
            return 0

    def cleanup_backup_folder(self) -> bool:
        """
        Remove the entire backups folder if it's empty after cleanup.
        """
        try:
            # Check if backups directory exists and is empty
            if self.backups_dir.exists():
                # List all files in backups directory
                remaining_files = list(self.backups_dir.glob("*"))
                if not remaining_files:
                    self.backups_dir.rmdir()
                    print(f"  Removed empty backups directory: {self.backups_dir}")
                    return True
                else:
                    print(
                        f" Backups directory not empty ({len(remaining_files)} files remain): {self.backups_dir}"
                    )
                    return False
            return False
        except Exception as e:
            print(f"Could not remove backups directory: {e}")
            return False

    def cleanup_all_backups(self) -> int:
        """
        Remove ALL backup files after successful training.
        """
        try:
            deleted_count = 0

            # Delete all .pth backup files
            for pth_file in self.backups_dir.glob("backup_*.pth"):
                pth_file.unlink(missing_ok=True)

                # Also delete corresponding .json file
                json_file = pth_file.with_suffix(".json")
                json_file.unlink(missing_ok=True)

                deleted_count += 1
                print(f"Deleted backup: {pth_file.name}")

            print(f"Deleted all {deleted_count} backup files after successful training")
            return deleted_count

        except Exception as e:
            print(f"Error during full backup cleanup: {e}")
            return 0

    def list_backups(self) -> list[dict[str, Any]]:
        backups = []

        for backup_path in self.backups_dir.glob("backup_*.pth"):
            info_path = backup_path.with_suffix(".json")

            if info_path.exists():
                try:
                    with open(info_path) as f:
                        backup_info = json.load(f)
                    backups.append(backup_info)
                except json.JSONDecodeError as e:
                    print(f"Error reading backup info {info_path}: {e}")
                    continue

        backups.sort(key=lambda x: x["epoch"], reverse=True)

        return backups

    def get_latest_backup(self) -> Optional[Path]:
        backups = self.list_backups()
        if not backups:
            return None

        latest_backup = backups[0]
        return Path(latest_backup["backup_path"])

    def restore_backup(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        backup_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        if backup_path is None:
            backup_path = self.get_latest_backup()

        if backup_path is None or not backup_path.exists():
            raise FileNotFoundError(f"No backup found at {backup_path}")

        checkpoint = torch.load(backup_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"    Restored optimizer state")
        if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            print(f"    Restored learning rate scheduler state")
        elif lr_scheduler is not None and "lr_scheduler_state_dict" not in checkpoint:
            print(f"    Note: Backup does not contain lr_scheduler state")

        print(f"    Restored from backup: {backup_path.name}")
        print(f"    Epoch: {checkpoint['epoch']}")
        print(f"    Val Acc: {checkpoint['val_acc']:.4f}")
        print(f"    Timestamp: {checkpoint['timestamp']}")
        
        if "additional_data" in checkpoint:
            print(f"    Backup contains additional data")

        return checkpoint

    def get_backup_info(self, backup_path: Path) -> Dict[str, Any]:
        info_path = backup_path.with_suffix(".json")
        
        if not info_path.exists():
            raise FileNotFoundError(f"No metadata found for backup: {backup_path}")
        
        with open(info_path) as f:
            return json.load(f)


def resume_training(
    run_folder: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    backup_path: Optional[Path] = None,
    **train_kwargs,
) -> tuple[torch.nn.Module, dict, Path]:
    backup_manager = BackupManager(run_folder)

    checkpoint = backup_manager.restore_backup(
        model=model, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        backup_path=backup_path
    )
    
    resumed_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {resumed_epoch}")
    
    if "num_epochs" in train_kwargs:
        print(f"Training will continue to epoch {train_kwargs['num_epochs']}")
    

    train_kwargs["starting_epoch"] = resumed_epoch

    # To avoid circular imports
    from .train import train_model

    return train_model(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **train_kwargs,
    )