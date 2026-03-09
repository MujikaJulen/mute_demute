"""Auto-encoder genérico basado en modelos de timm.

Esta implementación permite instanciar un auto-encoder pasando el nombre
(del string) de un modelo de timm y construyendo un decoder sencillo que
desempaqueta la representación latente a la resolución de entrada.

Ejemplo:
    from RedNeuronal.model import TimmAutoEncoder

    ae = TimmAutoEncoder("resnet18", pretrained=True, input_shape=(3, 100, 100))
    out = ae(torch.randn(1, 3, 100, 100))

"""

from __future__ import annotations
from transformers import ViTMAEConfig, ViTMAEForPreTraining
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderWithClassifier(nn.Module):

    def __init__(self, hidden_size, num_labels, freeze_encoder=False):
        super().__init__()
        config = ViTMAEConfig(
            image_size=224,
            patch_size=16,
            hidden_size=384,          
            num_hidden_layers=12,     
            num_attention_heads=6,    
            decoder_hidden_size=256,  
            mask_ratio=0.40           
        )
        mae = ViTMAEForPreTraining(config)
        ruta_pesos = "outs/pretrain/vit_small_patch16_224/autoencoder_best.pth"
        mae.load_state_dict(torch.load(ruta_pesos, map_location="cpu"))
        self.encoder = mae.vit
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//3), 
            nn.ReLU(),                                  
            nn.Linear(hidden_size//3, num_labels)   
        )

    def forward(self, input_ids):
        outputs = self.encoder(pixel_values=input_ids)
        hidden = outputs.last_hidden_state           
        pooled = hidden.mean(dim=1)             
        logits = self.classifier(pooled)             
        return logits