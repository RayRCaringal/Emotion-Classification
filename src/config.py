"""
Configuration settings for emotion classification project.
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Dataset
NUM_LABELS = 7
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Model
DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"

# Hyperparameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 5
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_STEPS = 500

# LoRA Hyperparameters
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LORA_TARGET_MODULES = ["query", "value"]
DEFAULT_LORA_LEARNING_RATE = 2e-4

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# DataLoader
NUM_WORKERS = 2
PIN_MEMORY = True if torch.cuda.is_available() else False

# Weights & Biases Configuration
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = "emotion-classification"
WANDB_DIR = "wandb"

# Linear Probe Hyperparameters
DEFAULT_LINEAR_PROBE_LEARNING_RATE = 1e-3
DEFAULT_LINEAR_PROBE_NUM_EPOCHS = 10
DEFAULT_LINEAR_PROBE_BATCH_SIZE = 64  