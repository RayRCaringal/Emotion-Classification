"""
Experiment metadata and configuration management.
"""

import json
import re
from datetime import datetime
from pathlib import Path

from .config import CHECKPOINTS_DIR


def get_next_folder(base_name: str, base_dir: Path = CHECKPOINTS_DIR) -> Path:
    """
    Get the next available folder
        Pattern: name -> name1 -> name2
    """
    pattern = re.compile(rf"{re.escape(base_name)}(\d*)$")

    # Find all existing folders with this base name
    existing_numbers = []
    for item in base_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                number_str = match.group(1)
                if number_str:  # Has a number
                    existing_numbers.append(int(number_str))
                else:  # No number
                    existing_numbers.append(0)

    if existing_numbers:
        next_number = max(existing_numbers) + 1
        # Only add number if it's not the first one
        if next_number == 1 and 0 in existing_numbers:
            run_folder = base_dir / f"{base_name}{next_number}"
        elif next_number > 0:
            run_folder = base_dir / f"{base_name}{next_number}"
        else:
            run_folder = base_dir / f"{base_name}"
    else:
        run_folder = base_dir / f"{base_name}"

    run_folder.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder.name}")

    return run_folder


##--------------------------------------
# Training Parameters
# --------------------------------------
def save_training_parameters(run_folder: Path, parameters: dict) -> None:
    """
    Save training parameters to JSON file.
    """
    params_path = run_folder / "training_parameters.json"

    parameters["timestamp"] = datetime.now().isoformat()

    with open(params_path, "w") as f:
        json.dump(parameters, f, indent=2, default=str)

    print(f"Training parameters saved to: {params_path}")


def load_training_parameters(run_folder: Path) -> dict:
    """
    Load training parameters from JSON file.
    """
    params_path = run_folder / "training_parameters.json"

    if not params_path.exists():
        raise FileNotFoundError(f"Training parameters not found: {params_path}")

    with open(params_path) as f:
        parameters = json.load(f)

    return parameters


##--------------------------------------
# Training History
# --------------------------------------
def load_training_history(run_folder: Path) -> dict:
    """
    Load training history from JSON file.

    Parameters
    ----------
    run_folder : Path
        Run folder containing history

    Returns
    -------
    dict
        Training history
    """
    history_path = run_folder / f"history_{run_folder.name}.json"

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        print(f"✅ Loaded history from: {history_path}")
        return history
    else:
        print(f"❌ History file not found: {history_path}")
        return {}


def save_training_history(run_folder: Path, history: dict) -> None:
    """
    Save training history to JSON file.

    Parameters
    ----------
    run_folder : Path
        Run folder to save history in
    history : dict
        Training history to save
    """
    history_path = run_folder / f"history_{run_folder.name}.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_path}")


##--------------------------------------
# Training Runs
# --------------------------------------
def find_latest_run_for_experiment(
    experiment_name: str, checkpoints_dir: Path = None
) -> Path:
    if checkpoints_dir is None:
        checkpoints_dir = CHECKPOINTS_DIR

    # Pattern: name -> name1 -> name2
    pattern = re.compile(rf"{re.escape(experiment_name)}(\d*)$")

    matching_folders = []

    for folder in checkpoints_dir.iterdir():
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                # Extract the number if it exists
                number_str = match.group(1)
                if number_str:
                    folder_number = int(number_str)
                else:
                    folder_number = 0  # original

                matching_folders.append((folder, folder_number))

    if not matching_folders:
        raise FileNotFoundError(f"No runs found for experiment: {experiment_name}")

    # Sort by folder number
    matching_folders.sort(key=lambda x: x[1], reverse=True)

    latest_folder = matching_folders[0][0]
    print(f" Latest run for '{experiment_name}': {latest_folder.name}")

    return latest_folder


def find_all_training_runs(checkpoints_dir: Path = None) -> list[dict]:
    if checkpoints_dir is None:
        checkpoints_dir = CHECKPOINTS_DIR

    runs = []

    for run_folder in checkpoints_dir.iterdir():
        if run_folder.is_dir():
            # Look for the best model checkpoint
            best_model_path = run_folder / f"best_{run_folder.name}.pth"

            if best_model_path.exists():
                runs.append(
                    {
                        "name": run_folder.name,
                        "folder": run_folder,
                        "best_model_path": best_model_path,
                        "history_path": run_folder / f"history_{run_folder.name}.json",
                        "params_path": run_folder / "training_parameters.json",
                    }
                )

    print(f"Found {len(runs)} training runs")
    return runs
