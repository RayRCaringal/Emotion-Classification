"""
ViT model class with LoRA support.
"""
from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import ViTForImageClassification
from transformers.modeling_outputs import ImageClassifierOutput
from datetime import datetime
from .base_model import BaseModel

class ViTModel(BaseModel):
    def __init__(
        self,
        num_labels: int = 7,
        model_name: str = "",
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(
            num_labels=num_labels,
            model_name=model_name,
            device=device,
        )

        print(f"Loading pre-trained ViT model: {model_name}")
        
        # Load HuggingFace model
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            **kwargs,
        )
        
        self.model.eval()
        
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        return self.model(pixel_values=pixel_values, labels=labels)

    def get_classifier_parameters(self) -> List[nn.Parameter]:
        if hasattr(self.model, "classifier"):
            return list(self.model.classifier.parameters())
        raise AttributeError(
            "Underlying ViT model does not have a 'classifier' attribute."
        )

    def get_encoder_parameters(self) -> List[nn.Parameter]:
        if hasattr(self.model, "vit"):
            return list(self.model.vit.parameters())
        raise AttributeError(
            "Model does not have a 'vit' attribute."
        )

    def get_underlying_model(self) -> nn.Module:
        return self.model

    def setup_for_lora(
        self, 
        lora_config: LoraConfig,
        target_modules: Optional[List[str]] = None
    ) -> PeftModel:
        """
        Parameters
        ----------
        target_modules : 
            Which attention modules to target. 
                Options: ["query", "value", "key", "dense"]
        
        """
        print(f"Setting up {self.model_name} for LoRA fine-tuning...")
        print(f"Config: r={lora_config.r}, "
                f"modules={lora_config.target_modules}")

        lora_model = get_peft_model(self, lora_config)
        
        # Unfreeze Classifier Head 
        for param in self.get_classifier_parameters():
            param.requires_grad = True
            
        self._current_training_mode = "lora"
        self._last_configured_at = datetime.now()
        
        print("✅ LoRA setup complete")
        lora_model.print_trainable_parameters() 
        
        return lora_model