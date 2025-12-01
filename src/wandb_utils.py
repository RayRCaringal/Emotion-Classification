"""
WandB utility functions for managing online/offline modes, syncing, and logging.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import torch
import wandb

from src.config import WANDB_API_KEY, WANDB_DIR

DEFAULT_MODE: Literal["online", "offline", "disabled"] = "offline"


def set_wandb_mode(mode: Literal["online", "offline", "disabled"] = DEFAULT_MODE):
    os.environ["WANDB_MODE"] = mode
    print(f"WandB mode set to: {mode.upper()}")


def get_wandb_mode() -> str:
    return os.environ.get("WANDB_MODE", "online")


def check_wandb_mode():
    current_mode = get_wandb_mode()
    print(f"Current WandB mode: {current_mode.upper()}")

    return current_mode


def login(
    project: str = "emotion-classifier",
    mode: Literal["online", "offline", "disabled"] = DEFAULT_MODE,
):
    set_wandb_mode(mode)
    wandb.login(key=WANDB_API_KEY)
    print(f"WandB initialized in {mode.upper()} mode for project: {project}")

    return check_wandb_mode()


def init_wandb_run(
    project: str = "emotion-classification",
    name: str | None = None,
    config: dict[str, Any] | None = None,
    model: torch.nn.Module = None,
):
    if get_wandb_mode() == "disabled":
        print("WandB is disabled")
        return None

    run = wandb.init(project=project, name=name, config=config)

    if model is not None:
        wandb.watch(model, log="all", log_freq=100)

    print(f"WandB run started: {run.name}")
    if run.url:
        print(f"WandB Dashboard: {run.url}")

    return run


def log_batch(loss: float, learning_rate: float, epoch: int, batch_idx: int):
    if get_wandb_mode() == "disabled" or wandb.run is None:
        return

    wandb.log(
        {
            "batch_train_loss": loss,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "batch": batch_idx,
        }
    )


def log_epoch(
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    train_precision: float,
    train_recall: float,
    train_f1: float,
    val_loss: float,
    val_accuracy: float,
    val_precision: float,
    val_recall: float,
    val_f1: float,
    learning_rate: float,
):
    if get_wandb_mode() == "disabled" or wandb.run is None:
        return

    wandb.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "learning_rate": learning_rate,
        }
    )


def sync_offline_runs():
    """
    Sync all offline WandB runs.
    """
    print("Syncing offline runs...")

    try:
        result = subprocess.run(
            ["wandb", "sync", "wandb/"], capture_output=True, text=True, check=True
        )
        print("Offline runs synced successfully!")
        if result.stdout:
            print("Sync output:", result.stdout)
        return True

    except Exception as e:
        print("Failed to Sync Offline Runs:")
        print(f"Unexpected Error: {e}")
        return False


def list_offline_runs():
    """List available offline runs that can be synced."""
    wandb_dir = Path(WANDB_DIR)
    if wandb_dir.exists():
        offline_runs = [d for d in os.listdir(WANDB_DIR) if d.startswith("offline-run")]
        print(f"Found {len(offline_runs)} offline run(s):")
        for run in offline_runs:
            run_path = wandb_dir / run
            size_mb = get_folder_size_mb(run_path)
            print(f"  - {run} ({size_mb:.1f} MB)")
        return offline_runs
    else:
        print(f"{WANDB_DIR} directory not found")
        return []


def clear_offline_runs(confirm: bool = False) -> int:
    """
    Parameters
    ----------
    confirm :
        Safety flag - must be True to actually delete files
    """
    wandb_dir = Path(WANDB_DIR)

    if not wandb_dir.exists():
        print("No wandb directory found - nothing to clear")
        return 0

    offline_runs = [d for d in os.listdir(WANDB_DIR) if d.startswith("offline-run")]

    if not offline_runs:
        print("No offline runs found - nothing to clear")
        return 0

    print(f"Found {len(offline_runs)} offline run(s) to delete:")

    total_size = 0
    for run in offline_runs:
        run_path = wandb_dir / run
        size_mb = get_folder_size_mb(run_path)
        total_size += size_mb
        print(f"   - {run} ({size_mb:.1f} MB)")

    print(f"Total size: {total_size:.1f} MB")

    if not confirm:
        print("This is a dry run. Set confirm=True to delete")
        return 0

    # Actually delete the files
    deleted_count = 0
    for run in offline_runs:
        run_path = wandb_dir / run
        try:
            shutil.rmtree(run_path)
            print(f"✅ Deleted: {run}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ Failed to delete {run}: {e}")

    print(f"🎯 Successfully deleted {deleted_count} offline run(s)")

    # Try to remove the wandb directory if it's empty
    try:
        if not any(wandb_dir.iterdir()):
            wandb_dir.rmdir()
            print(f"Removed empty wandb directory: {wandb_dir}")
    except:
        pass  # Directory not empty or other issue

    return deleted_count


def get_folder_size_mb(folder_path: Path) -> float:
    """
    Calculate the size of a folder in MB.

    Parameters
    ----------
    folder_path : Path
        Path to the folder

    Returns
    -------
    float
        Size in megabytes
    """
    try:
        total_size = 0
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    except:
        return 0.0
