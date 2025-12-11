"""
CNN model class that wraps the HuggingFace ResNetForImageClassification 
(using AutoModel) model and inherits from BaseModel for standardized training logic.
"""
from typing import Any, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification
from transformers.modeling_outputs import ImageClassifierOutput 

from .base_model import BaseModel
from src.types import ModelMetadata


class CNNModel(BaseModel):
    """
    A wrapper around HuggingFace's AutoModelForImageClassification (ResNet-18) 
    model, implementing the BaseModel interface.
    """

    def __init__(
        self,
        num_labels: int = 7,
        model_name: str = "microsoft/resnet-18",
        device: Optional[torch.device] = None,
        # Allow passing through arguments for AutoModelForImageClassification.from_pretrained
        **kwargs,
    ):
        # We pass the ResNet-18 name to the parent
        super().__init__(
            num_labels=num_labels,
            model_name=model_name,
            device=device,
        )

        print(f"Loading pre-trained CNN model: {model_name}")
        
        # Instantiate the actual HuggingFace model and store it in self._model
        # AutoModelForImageClassification for ResNet uses ResNetForImageClassification
        self._model: AutoModelForImageClassification = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            **kwargs,
        )
        
        # Move the internal model to device
        self._model.to(self.device)
        self.to_device(self.device)

        # Set initial training mode (default to full fine-tune)
        self.setup_for_full_finetune()
        
    def get_classifier_parameters(self) -> List[nn.Parameter]:
        """
        Returns the parameters of the classification head (model.classifier).
        
        In the ResNetForImageClassification structure (used by AutoModel), 
        the final layer is called 'classifier'.
        """
        if hasattr(self._model, "classifier"):
            return list(self._model.classifier.parameters())
        
        raise AttributeError(
            f"Underlying model {type(self._model).__name__} does not have a 'classifier' attribute."
        )

    def get_encoder_parameters(self) -> List[nn.Parameter]:
        """
        Returns the parameters of the encoder backbone (ResNet feature layers).
        
        In the ResNetForImageClassification structure, the feature extractor/backbone 
        is stored in the top-level attribute (often matching the model name's backbone).
        For ResNet, this is often the 'resnet' attribute itself, or the top level 
        minus the classifier.
        
        We will iterate over named parameters and exclude the 'classifier'.
        """
        encoder_params: List[nn.Parameter] = []
        for name, param in self._model.named_parameters():
            if not name.startswith("classifier."):
                encoder_params.append(param)
        
        if not encoder_params:
             raise AttributeError("Could not find any encoder parameters.")
             
        return encoder_params



