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
from transformers import ViTMAEForPreTraining
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderWithClassifier(nn.Module):

    def init(self, encoder, hidden_size, num_labels, freeze_encoder=True):
        super().init()
        mae = ViTMAEForPreTraining.from_pretrained(".outs/pretrain/vit_small_patch16_224/autoencoder_best.pth")
        encoder = mae.vit
        self.encoder = encoder
        for param in self.vit.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state           
        pooled = hidden.mean(dim=1)                  
        logits = self.classifier(pooled)             
        return logits