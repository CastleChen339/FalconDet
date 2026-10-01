import torch
import torch.nn as nn
from ...core import register


@register()
class Backbone3D(nn.Module):
    """
    3D backbone wrapper around pytorchvideo TorchHub models.

    Args:
        name: Backbone model name.
        return_ids: Block indices to return as feature maps.
        pretrained: Whether to load pretrained weights.
    """

    def __init__(self, name: str = "i3d_r50", return_ids=None, pretrained: bool = True):
        super().__init__()

        allowed_models = [
            "c2d_r50",
            "i3d_r50",
            "slow_r50",
            "slowfast_r50",
            "slowfast_r101",
            "slowfast_16x8_r101_50_50",
            "csn_r101",
            "r2plus1d_r50",
            "x3d_xs",
            "x3d_s",
            "x3d_m",
            "x3d_l"
        ]

        if name not in allowed_models:
            raise ValueError(
                f"Invalid model name '{name}'. Must be one of: {', '.join(allowed_models)}"
            )

        self.name = name
        self.backbone = torch.hub.load(
            repo_or_dir="./pytorchvideo/facebookresearch_pytorchvideo_main",
            model=name,
            source='local',
            pretrained=pretrained
        )

        self.return_ids = [1, 2, 3] if return_ids is None else return_ids

    def forward(self, x):
        """
        Forward pass through backbone blocks.

        Args:
            x: Input tensor [B, C, T, H, W].

        Returns:
            List of feature tensors for requested blocks.
        """
        feats = []
        blocks = self.backbone.blocks

        for i in range(len(blocks) - 1):
            x = blocks[i](x)
            if i in self.return_ids:
                feats.append(x)

        return feats
