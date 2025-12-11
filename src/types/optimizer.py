"""
Optimizer configuration types.
"""

from typing import Optional, Tuple
from enum import StrEnum 

from pydantic import BaseModel, Field

class OptimizerType(StrEnum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"

class OptimizerConfig(BaseModel):

    optimizer_type: OptimizerType = Field(
        default=OptimizerType.ADAMW, description="Type of optimizer"
    )
    learning_rate: float = Field(gt=0, description="Learning rate")
    weight_decay: float = Field(ge=0, default=0.01, description="Weight decay")
    betas: Tuple[float, float] = Field(
        default=(0.9, 0.999), description="Beta coefficients for Adam/AdamW"
    )
    momentum: Optional[float] = Field(
        default=None, ge=0, le=1, description="Momentum for SGD"
    )
    eps: float = Field(default=1e-8, gt=0, description="Epsilon for numerical stability")

    class Config:
        frozen = True