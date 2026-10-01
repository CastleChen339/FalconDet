import torch.nn as nn

from ...core import register

__all__ = [
    "FalconDet",
]


@register()
class FalconDet(nn.Module):
    """
    FalconDet backbone-encoder-decoder wrapper.
    """

    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        """
        Args:
            backbone: Feature extraction backbone.
            encoder: Feature encoder.
            decoder: Detection decoder.

        Returns:
            None.
        """
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        """
        Forward pass through backbone, encoder, and decoder.

        Args:
            x: Input tensor [B, C, T, H, W].
            targets: Optional target list for training.

        Returns:
            Dict of model outputs.
        """
        x = self.backbone(x)

        x = self.encoder(x)
        vox_density = x['vox_density']

        x = self.decoder(x['outs'], targets)
        x['vox_density'] = vox_density

        return x

