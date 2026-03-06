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

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _create_timm_encoder(model_name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    """Crea un encoder a partir de un modelo timm.

    Se intenta crear el modelo en modo `features_only` (salidas de mapas de
    características). Si el modelo no soporta `features_only`, se crea con
    `num_classes=0` para eliminar la cabeza de clasificación.
    """

    try:
        # Muchos modelos timm soportan `features_only=True`.
        return timm.create_model(model_name, pretrained=pretrained, features_only=True, **kwargs)
    except TypeError:
        # Algunos modelos no soportan features_only, en ese caso pedimos que devuelva
        # la salida previa a la cabeza de clasificación.
        return timm.create_model(model_name, pretrained=pretrained, num_classes=0, **kwargs)


class TimmAutoEncoder(nn.Module):
    """Autoencoder genérico usando un encoder timm y un decoder simple.

    Args:
        model_name: Nombre del modelo en timm (por ejemplo: "resnet18", "efficientnet_b0").
        pretrained: Si se cargan pesos preentrenados.
        input_shape: Tupla (C, H, W) con la resolución de entrada que se usará para
            construir el decoder.
        decoder_channels: Lista de canales intermedios para el decoder. Si es None,
            se usa una secuencia predeterminada.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        input_shape: tuple[int, int, int] = (3, 224, 224),
        decoder_channels: list[int] | None = None,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.encoder = _create_timm_encoder(model_name, pretrained=pretrained)

        # Determinar la forma de la salida del encoder (canales, H, W)
        # Algunos encoders devuelven una lista de mapas de características;
        # tomamos el último (más profundo).
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            encoded = self.encoder(dummy)
        if isinstance(encoded, (list, tuple)):
            encoded = encoded[-1]
        if encoded.dim() != 4:
            raise ValueError(
                f"Se esperaba un mapa de características 4D del encoder, se obtuvo: {encoded.shape}"
            )
        self._enc_channels, self._enc_h, self._enc_w = encoded.shape[1:]

        # Decoder simple: primero escalamos a la resolución de entrada y luego
        # aplicamos convoluciones para refinar la reconstrucción.
        if decoder_channels is None:
            decoder_channels = [self._enc_channels // 2, 64, 32]
        self.decoder = nn.Sequential(
            nn.Upsample(size=(input_shape[1], input_shape[2]), mode="bilinear", align_corners=False),
            nn.Conv2d(self._enc_channels, decoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[1], decoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[2], input_shape[0], kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pasa el tensor `x` por el encoder y decoder.

        Devuelve la reconstrucción con el mismo tamaño que la entrada.
        """
        feats = self.encoder(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        out = self.decoder(feats)
        if out.shape[2:] != x.shape[2:]:
            # Aseguramos que la salida tenga el mismo tamaño espacial que la entrada.
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


if __name__ == "__main__":
    # Ejemplo mínimo de uso.
    model = TimmAutoEncoder("resnet18", pretrained=False, input_shape=(3, 100, 100))
    x = torch.randn(2, 3, 100, 100)
    y = model(x)
    print("input:", x.shape, "output:", y.shape)
