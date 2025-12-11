"""
Parameter group information types.
"""
from typing import List, Optional
from enum import StrEnum 

from pydantic import BaseModel, Field


class ParameterGroupType(StrEnum):
    ENCODER = "encoder"
    CLASSIFIER = "classifier"
    EMBEDDINGS = "embeddings"
    ATTENTION = "attention"
    ALL = "all"
    CUSTOM = "custom"


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