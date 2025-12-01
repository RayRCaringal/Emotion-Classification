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
    """Decorator to retry WandB operations on connection errors"""

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


def sync_offline_runs(run_path: str | Path | None = None, all_runs: bool = False):
    """
    Sync offline WandB runs to the cloud.

    Parameters
    ----------
    run_path : str | Path | None
        Specific run to sync (e.g., "wandb/offline-run-20231201_123456-abc123")
        If None and all_runs=False, will list available runs
    all_runs : bool
        If True, sync all offline runs in the WANDB_DIR

    Returns
    -------
    bool
        True if sync succeeded, False otherwise

    Examples
    --------
    # List available runs
    >>> sync_offline_runs()

    # Sync all runs
    >>> sync_offline_runs(all_runs=True)

    # Sync specific run
    >>> sync_offline_runs("wandb/offline-run-20231201_123456-abc123")
    """
    wandb_dir = Path(WANDB_DIR)

    # Check if wandb directory exists
    if not wandb_dir.exists():
        print(f"❌ WandB directory not found: {wandb_dir}")
        return False

    # Get list of offline runs
    offline_runs = [
        d
        for d in wandb_dir.iterdir()
        if d.is_dir() and d.name.startswith("offline-run")
    ]

    if not offline_runs:
        print("No offline runs found to sync")
        return False

    # If no specific run and not syncing all, just list available runs
    if run_path is None and not all_runs:
        print(f"Found {len(offline_runs)} offline run(s) to sync:")
        for i, run in enumerate(offline_runs, 1):
            size_mb = get_folder_size_mb(run)
            print(f"  {i}. {run.name} ({size_mb:.1f} MB)")
        print("\nTo sync:")
        print("  - All runs: sync_offline_runs(all_runs=True)")
        print("  - Most recent: sync_latest_offline_run()")
        print(f"  - Specific run: sync_offline_runs('{offline_runs[0]}')")
        return False

    # Determine what to sync
    if all_runs:
        sync_targets = offline_runs
        print(f"Syncing all {len(sync_targets)} offline runs...")
    else:
        sync_target = Path(run_path)
        if not sync_target.exists():
            print(f"❌ Run path not found: {sync_target}")
            return False
        sync_targets = [sync_target]
        print(f"Syncing run: {sync_target.name}")

    # Sync each target
    success_count = 0
    fail_count = 0

    for target in sync_targets:
        try:
            print(f"\n  Syncing: {target.name}...")
            result = subprocess.run(
                ["wandb", "sync", str(target)],
                capture_output=True,
                text=True,
                check=True,
                timeout=300,  # 5 minute timeout per run
            )

            if result.returncode == 0:
                print(f"  ✅ Synced: {target.name}")
                success_count += 1

                # Show output if there's useful info
                if result.stdout and "View run at:" in result.stdout:
                    # Extract and show the URL
                    for line in result.stdout.split("\n"):
                        if "View run at:" in line or "https://" in line:
                            print(f"     {line.strip()}")
            else:
                print(f"  ⚠️  Warning: {target.name} returned code {result.returncode}")
                if result.stderr:
                    print(f"     Error: {result.stderr}")
                fail_count += 1

        except subprocess.TimeoutExpired:
            print(f"  ❌ Timeout syncing: {target.name}")
            fail_count += 1

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to sync: {target.name}")
            print(f"     Return code: {e.returncode}")
            if e.stderr:
                print(f"     Error: {e.stderr}")
            fail_count += 1

        except FileNotFoundError:
            print("  ❌ 'wandb' command not found. Is wandb installed?")
            print("     Install with: pip install wandb")
            return False

        except Exception as e:
            print(f"  ❌ Unexpected error syncing {target.name}: {e}")
            fail_count += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"Sync complete: {success_count} succeeded, {fail_count} failed")
    print("=" * 70)

    return fail_count == 0


def sync_latest_offline_run():
    """
    Sync only the most recent offline run.
    Useful for syncing after a single training session.

    Returns
    -------
    bool
        True if sync succeeded, False otherwise
    """
    wandb_dir = Path(WANDB_DIR)

    if not wandb_dir.exists():
        print(f"❌ WandB directory not found: {wandb_dir}")
        return False

    # Get all offline runs sorted by modification time (most recent first)
    offline_runs = [
        d
        for d in wandb_dir.iterdir()
        if d.is_dir() and d.name.startswith("offline-run")
    ]

    if not offline_runs:
        print("No offline runs found to sync")
        return False

    # Sort by modification time
    latest_run = max(offline_runs, key=lambda p: p.stat().st_mtime)

    print(f"Latest offline run: {latest_run.name}")
    return sync_offline_runs(str(latest_run))


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
