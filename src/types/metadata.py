"""
Model metadata and capabilities types.
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from src.types.model import TrainingMode


class InputSpec(BaseModel):
    """Input requirements and preprocessing specifications"""

    image_size: Tuple[int, int] = Field(description="Expected input image size (H, W)")
    channels: int = Field(ge=1, le=4, description="Number of input channels")
    normalization_mean: Tuple[float, float, float] = Field(
        description="Mean values for normalization (R, G, B)"
    )
    normalization_std: Tuple[float, float, float] = Field(
        description="Std values for normalization (R, G, B)"
    )
    pixel_value_range: Tuple[float, float] = Field(
        default=(0.0, 1.0), description="Expected range of pixel values"
    )

    class Config:
        frozen = True


class ModelCapabilities(BaseModel):
    """Capabilities and features supported by the model"""

    supports_attention_extraction: bool = Field(
        default=False, description="Can extract attention weights"
    )
    supports_feature_extraction: bool = Field(
        default=False, description="Can extract intermediate features"
    )
    supports_layer_freezing: bool = Field(
        default=False, description="Can freeze individual layers"
    )
    supports_discriminative_finetuning: bool = Field(
        default=False, description="Supports different learning rates per layer"
    )
    supported_training_modes: List[TrainingMode] = Field(
        description="List of supported training modes"
    )

    class Config:
        frozen = True


class ModelMetadata(BaseModel):
    
    architecture: str = Field(description="Architecture class name (e.g., 'ViTModel')")
    base_model: str = Field(description="Base model identifier (e.g., 'google/vit-base-patch16-224')")
    num_labels: int = Field(description="Number of output labels")
    num_parameters: Optional[int] = Field(default=None, description="Total parameters")
    
    class Config:
        frozen = True