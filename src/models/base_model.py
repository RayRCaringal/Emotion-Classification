"""
Base model class for all emotion classification models.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn as nn

from src.types import (
    FreezingStrategy,
    ParameterCounts,
    ParameterGroupInfo,
    ParameterGroupType,
    TrainingMode,
)


class BaseModel(ABC, nn.Module): 
    def __init__(
        self,
        num_labels: int = 7,
        model_name: str = "base_model",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._current_training_mode: TrainingMode = "custom"
        self._last_configured_at: Optional[datetime] = None


    @abstractmethod
    def get_classifier_parameters(self) -> List[nn.Parameter]:
        """Returns parameters of the classification head."""
        pass

    @abstractmethod
    def get_encoder_parameters(self) -> List[nn.Parameter]:
        """Returns parameters of the encoder/backbone."""
        pass

    @abstractmethod
    def get_underlying_model(self) -> nn.Module:
        """
        Returns the underlying PyTorch model for PEFT operations.
        """
        pass

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all parameters where requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> ParameterCounts:
        """Count total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return ParameterCounts(
            total=total,
            trainable=trainable,
            frozen=frozen,
        )

    def get_parameter_group_info(
        self, group_type: ParameterGroupType
    ) -> ParameterGroupInfo:
        """Get information about a specific parameter group."""
        if group_type == "classifier":
            params = self.get_classifier_parameters()
            description = "Classifier head parameters"
        elif group_type == "encoder":
            params = self.get_encoder_parameters()
            description = "Encoder/backbone parameters"
        elif group_type == "all":
            params = list(self.parameters())
            description = "All model parameters"
        else:
            raise ValueError(f"Unsupported parameter group type: {group_type}")

        num_params = sum(p.numel() for p in params)
        is_frozen = all(not p.requires_grad for p in params) if params else True

        return ParameterGroupInfo(
            group_type=group_type,
            num_parameters=num_params,
            is_frozen=is_frozen,
            description=description,
        )

    def freeze_encoder(self) -> None:
        """Freeze encoder, keep classifier trainable (for linear probe)."""
        for param in self.get_encoder_parameters():
            param.requires_grad = False

        for param in self.get_classifier_parameters():
            param.requires_grad = True

        self._print_trainable_params()

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (for full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

        self._print_trainable_params()

    def get_freezing_strategy(self) -> FreezingStrategy:
        """Get current freezing strategy description."""
        encoder_info = self.get_parameter_group_info("encoder")
        classifier_info = self.get_parameter_group_info("classifier")

        frozen_components: List[ParameterGroupType] = []
        trainable_components: List[ParameterGroupType] = []

        if encoder_info.is_frozen:
            frozen_components.append("encoder")
        else:
            trainable_components.append("encoder")

        if classifier_info.is_frozen:
            frozen_components.append("classifier")
        else:
            trainable_components.append("classifier")

        # Generate description
        if self._current_training_mode == "lora":
            description = "LoRA adapters + Classifier trainable"
        elif not frozen_components:
            description = "All parameters trainable (Full Fine-tune)"
        elif not trainable_components:
            description = "All parameters frozen"
        else:
            frozen_str = ', '.join(frozen_components)
            trainable_str = ', '.join(trainable_components)
            description = f"Frozen: {frozen_str}; Trainable: {trainable_str}"

        return FreezingStrategy(
            frozen_components=frozen_components,
            frozen_layer_indices=[],
            trainable_components=trainable_components,
            description=description,
        )


    def setup_for_linear_probe(self) -> None:
        """Configure for linear probe: frozen encoder, trainable classifier."""
        print(f"Setting up {self.model_name} for linear probe...")
        self.freeze_encoder()
        self._current_training_mode = "linear_probe"
        self._last_configured_at = datetime.now()
        self._print_trainable_params()

    def setup_for_full_finetune(self) -> None:
        """Configure for full fine-tuning: all parameters trainable."""
        print(f"Setting up {self.model_name} for full fine-tuning...")
        self.unfreeze_all()
        self._current_training_mode = "full_finetune"
        self._last_configured_at = datetime.now()
        self._print_trainable_params()

        
    def _print_trainable_params(self) -> None:
        """Print trainable parameter statistics."""
        counts = self.count_parameters()
        print(f"   Trainable: {counts.trainable:,} / {counts.total:,} "
              f"({counts.percentage_trainable:.2f}%)")

    def print_summary(self) -> None:
        """Print a formatted summary of the model configuration."""
        counts = self.count_parameters()
        strategy = self.get_freezing_strategy()
        
        print(f"\n{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"{'='*70}")
        print(f"Architecture: {type(self).__name__}")
        print(f"Training Mode: {self._current_training_mode}")
        print(f"Strategy: {strategy.description}")
        print(f"Labels: {self.num_labels}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {counts.total:,}")
        print(f"Trainable: {counts.trainable:,} ({counts.percentage_trainable:.2f}%)")
        print(f"Frozen: {counts.frozen:,}")
        print(f"{'='*70}\n")

    def to_device(self, device: Optional[torch.device] = None) -> "BaseModel":
        """
        Move model to device.
        
        Returns self for method chaining.
        """
        target_device = device or self.device
        self.to(target_device)
        self.device = target_device
        return self

    def get_checkpoint_metadata(self) -> dict:
        """Get metadata for checkpoint saving."""
        return {
            "architecture": type(self).__name__,
            "base_model_name": self.model_name,
            "num_labels": self.num_labels,
            "training_mode": self._current_training_mode,
            "total_parameters": self.count_parameters().total,
        }