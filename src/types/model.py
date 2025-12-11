"""
Model configuration, state, summary, and parameter types.
"""

from datetime import datetime
from typing import List, Optional
from enum import StrEnum

from pydantic import BaseModel, Field


class TrainingType(StrEnum):
    """'Either Full Fine Tuning or Linear Probe'"""
    FULL_FINETUNE = "full_finetune"
    LINEAR_PROBE = "linear_probe"

class TrainingMode(StrEnum):
    """Defines the specific training mode to use."""
    FULL_FINETUNE = "full_finetune"
    LINEAR_PROBE = "linear_probe"
    FROZEN = "frozen"
    CUSTOM = "custom"

class ParameterGroupType(StrEnum):
    ENCODER = "encoder"
    CLASSIFIER = "classifier"
    EMBEDDINGS = "embeddings"
    ATTENTION = "attention"
    ALL = "all"
    CUSTOM = "custom"

class ModelConfig(BaseModel):
    """Model Configuration"""

    num_labels: int = Field(default=7, ge=1, description="Number of output classes")
    model_name: str = Field(default="model", description="Name identifier for the model")
    device: str = Field(default="cuda", description="Device to place model on")

    class Config:
        frozen = False


class ViTConfig(ModelConfig):
    pretrained_model_name: str = Field(
        description="HuggingFace model identifier",
    )
    ignore_mismatched_sizes: bool = Field(
        default=True, description="Ignore size mismatches when loading pretrained weights"
    )


class HyperparametersConfig(BaseModel):

    learning_rate: float = Field(gt=0, description="Learning rate")
    batch_size: int = Field(ge=1, description="Batch size")
    num_epochs: int = Field(ge=1, description="Number of training epochs")
    weight_decay: float = Field(ge=0, description="Weight decay for regularization")
    warmup_steps: int = Field(default=500, ge=0, description="Number of warmup steps")

    class Config:
        frozen = False

class ParameterCounts(BaseModel):

    total: int = Field(ge=0, description="Total number of parameters")
    trainable: int = Field(ge=0, description="Number of trainable parameters")
    frozen: int = Field(ge=0, description="Number of frozen parameters")

    @property
    def percentage_trainable(self) -> float:
        """Calculate percentage of trainable parameters."""
        if self.total == 0:
            return 0.0
        return 100 * self.trainable / self.total

    class Config:
        frozen = True


class ModelSummary(BaseModel):

    model_name: str = Field(description="Model name identifier")
    num_labels: int = Field(ge=1, description="Number of output classes")
    device: str = Field(description="Device model is on")
    total_parameters: int = Field(ge=0, description="Total number of parameters")
    trainable_parameters: int = Field(ge=0, description="Number of trainable parameters")
    frozen_parameters: int = Field(ge=0, description="Number of frozen parameters")
    percentage_trainable: float = Field(
        ge=0, le=100, description="Percentage of trainable parameters"
    )

    class Config:
        frozen = True


class ParameterGroupInfo(BaseModel):

    group_type: ParameterGroupType = Field(description="Type of parameter group")
    num_parameters: int = Field(ge=0, description="Number of parameters in group")
    is_frozen: bool = Field(description="Whether parameters are frozen")
    layer_indices: Optional[List[int]] = Field(
        default=None, description="Layer indices if applicable"
    )
    description: str = Field(
        default="", description="Human-readable description of parameter group"
    )

    class Config:
        frozen = True


class FreezingStrategy(BaseModel):
    """What components are frozen in the model."""


    frozen_components: List[ParameterGroupType] = Field(
        default_factory=list, description="List of frozen component types"
    )
    frozen_layer_indices: List[int] = Field(
        default_factory=list, description="Indices of frozen layers"
    )
    trainable_components: List[ParameterGroupType] = Field(
        default_factory=list, description="List of trainable component types"
    )
    description: str = Field(
        default="", description="Human-readable description of freezing strategy"
    )

    class Config:
        frozen = True


class ModelState(BaseModel):

    training_mode: TrainingMode = Field(description="Current training mode")
    freezing_strategy: FreezingStrategy = Field(
        description="Description of what's frozen"
    )
    parameter_counts: ParameterCounts = Field(
        description="Current parameter statistics"
    )
    configured_at: Optional[datetime] = Field(
        default=None, description="When the model was configured"
    )
    is_ready_for_training: bool = Field(
        default=True, description="Whether model is ready for training"
    )

    class Config:
        frozen = True