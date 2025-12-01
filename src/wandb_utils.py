"""
WandB utility functions for managing online/offline modes, syncing, and logging.
"""

import os
import shutil
import subprocess
import time
from functools import wraps
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


def retry_on_connection_error(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionResetError, ConnectionError, OSError) as e:
                    if attempt < max_retries - 1:
                        print(
                            f"  WandB connection error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f" WandB connection failed after {max_retries} attempts: {e}"
                        )
                        # Don't raise - just continue without logging
                        return None
            return func(*args, **kwargs)

        return wrapper

    return decorator


def init_wandb_run(
    project: str = "emotion-classification",
    name: str | None = None,
    config: dict[str, Any] | None = None,
    model: torch.nn.Module = None,
):
    if get_wandb_mode() == "disabled":
        print("WandB is disabled")
        return None

    try:
        run = wandb.init(
            project=project,
            name=name,
            config=config,
            reinit=True,  # Allow multiple runs in same process
            settings=wandb.Settings(
                start_method="thread",  # Better for notebooks
                _disable_stats=True,  # Reduce system monitoring overhead
                _disable_meta=True,  # Reduce metadata collection
            ),
        )

        if model is not None:
            # Reduced log frequency from 100 to 500
            wandb.watch(model, log="all", log_freq=500)

        print(f"WandB run started: {run.name}")
        if run.url:
            print(f"WandB Dashboard: {run.url}")

        return run
    except Exception as e:
        print(f"⚠️  Failed to initialize WandB run: {e}")
        return None


@retry_on_connection_error()
def log_batch(loss: float, learning_rate: float, epoch: int, batch_idx: int):
    """Log batch-level training metrics (loss only)"""
    if get_wandb_mode() == "disabled" or wandb.run is None:
        return

    wandb.log(
        {
            "train/batch_loss": loss,
            "train/learning_rate": learning_rate,
            "epoch": epoch,
            "batch": batch_idx,
        }
    )


@retry_on_connection_error()
def log_epoch(
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    val_f1: float,
    val_precision: float | None = None,
    val_recall: float | None = None,
    learning_rate: float | None = None,
):
    """
    Log epoch-level metrics.

    Training: Only loss (optimization metric)
    Validation: Loss, Accuracy, F1, and optionally Precision/Recall
    """
    if get_wandb_mode() == "disabled" or wandb.run is None:
        return

    log_dict = {
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/accuracy": val_accuracy,
        "val/f1": val_f1,
    }

    # Optional metrics
    if val_precision is not None:
        log_dict["val/precision"] = val_precision
    if val_recall is not None:
        log_dict["val/recall"] = val_recall
    if learning_rate is not None:
        log_dict["train/learning_rate"] = learning_rate

    wandb.log(log_dict)


def finish_wandb_run(quiet: bool = False):
    if wandb.run is not None:
        try:
            wandb.finish(quiet=quiet)
            if not quiet:
                print("✅ WandB run finished successfully")
        except Exception as e:
            if not quiet:
                print(f"⚠️  Error finishing WandB run: {e}")
    else:
        if not quiet:
            print("No active WandB run to finish")


# In an attempt to resolve the Connectione Timeout Errors
def cleanup_wandb_run():
    try:
        if wandb.run is not None:
            wandb.finish(quiet=True)

        # Small delay to ensure cleanup completes
        time.sleep(0.5)

        print("WandB run cleaned up")
        return True
    except Exception as e:
        print(f"Error during WandB cleanup: {e}")
        return False


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
    try:
        total_size = 0
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    except:
        return 0.0
