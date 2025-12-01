"""
Creates Backup checkpoints in case of Training Failure.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

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
        is_final: bool = False,
    ) -> Path:
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

        torch.save(checkpoint, backup_path)

        # Backup JSON
        backup_info = {
            "backup_path": str(backup_path),
            "epoch": epoch,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "timestamp": checkpoint["timestamp"],
            "backup_type": backup_type,
            "is_final": is_final,
        }

        info_path = backup_path.with_suffix(".json")
        with open(info_path, "w") as f:
            json.dump(backup_info, f, indent=2)

        return backup_path

    def cleanup_old_backups(self):
        """
        Remove old backups beyond the maximum allowed.
        Keeps the most recent backups.
        """
        backup_files = list(self.backups_dir.glob("backup_*.pth"))

        if len(backup_files) <= self.max_backups:
            return

        # Sort by time
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        backups_to_delete = backup_files[self.max_backups :]

        for backup_path in backups_to_delete:
            backup_path.unlink(missing_ok=True)
            info_path = backup_path.with_suffix(".json")
            info_path.unlink(missing_ok=True)

            print(f"Deleted old backup: {backup_path.name}")

    def list_backups(self) -> list[dict[str, Any]]:
        backups = []

        for backup_path in self.backups_dir.glob("backup_*.pth"):
            info_path = backup_path.with_suffix(".json")

            # Shouldn't be an issue, but just in case filter for json
            if info_path.exists():
                with open(info_path) as f:
                    backup_info = json.load(f)
                backups.append(backup_info)

        backups.sort(key=lambda x: x["epoch"], reverse=True)

        return backups

    def get_latest_backup(self) -> Path | None:
        backups = self.list_backups()
        if not backups:
            return None

        latest_backup = backups[0]
        return Path(latest_backup["backup_path"])

    def restore_backup(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        backup_path: Path | None = None,
    ) -> dict[str, Any]:
        if backup_path is None:
            backup_path = self.get_latest_backup()

        if backup_path is None or not backup_path.exists():
            raise FileNotFoundError(f"No backup found at {backup_path}")

        checkpoint = torch.load(backup_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"    Restored from backup: {backup_path.name}")
        print(f"    Epoch: {checkpoint['epoch']}")
        print(f"    Val Acc: {checkpoint['val_acc']:.4f}")
        print(f"    Timestamp: {checkpoint['timestamp']}")

        return checkpoint


def resume_training(
    run_folder: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    backup_path: Path | None = None,
    **train_kwargs,
) -> tuple[torch.nn.Module, dict, Path]:
    backup_manager = BackupManager(run_folder)

    checkpoint = backup_manager.restore_backup(
        model=model, optimizer=optimizer, backup_path=backup_path
    )
    resumed_epoch = checkpoint["epoch"] + 1
    if "num_epochs" in train_kwargs:
        print(f" Resuming from epoch {resumed_epoch} to {train_kwargs['num_epochs']}")

    # to avoid circular imports
    from .train import train_model

    return train_model(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **train_kwargs,
    )
