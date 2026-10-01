import torch
import torch.nn as nn

class DepthwiseSeparableConv3D(nn.Module):
    """
    3D depthwise separable convolution block.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        dilation: Convolution dilation.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=False
        )
        self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ECA(nn.Module):
    """
    Efficient channel attention applied after depth pooling.

    Args:
        channels: Number of channels.
        k_size: Kernel size for 1D convolution.
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = y.unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial attention block (2D, CBAM-style).

    Args:
        kernel_size: Kernel size for attention convolution.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


class MSBlock3D(nn.Module):
    """
    Multi-scale parallel 3D convolution block.

    Args:
        channels: Number of channels.
        dilations: Dilation values for each branch.
    """
    def __init__(self, channels, dilations=(1,2,3)):
        super().__init__()
        self.branches = nn.ModuleList([
            DepthwiseSeparableConv3D(channels, channels, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = 0
        for b in self.branches:
            out = out + b(x)
        return self.fuse(out)


class VoxGen(nn.Module):
    """
    Voxel-based density generator.

    Args:
        in_channels: Input channels.
        channels: Hidden channels.
        num_blocks: Number of multi-scale blocks.
        dilations: Dilation values for each block.
    """
    def __init__(self, in_channels=64, channels=128, num_blocks=3, dilations=(1,2,3)):
        super().__init__()

        # Initial channel projection
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

        # Multi-scale blocks
        self.body = nn.Sequential(*[MSBlock3D(channels, dilations) for _ in range(num_blocks)])

        # Residual connection
        self.res_proj = nn.Identity()

        # 2D attention modules after depth pooling
        self.eca = ECA(channels)
        self.spatial_att = SpatialAttention()

        # Density head: [B, 1, H, W]
        self.density_head = nn.Sequential(
            nn.Conv2d(channels, channels//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Regression head: global intensity estimate
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor with shape [B, C, D, H, W].

        Returns:
            Tuple of (density_map, regression_score).
        """
        x = self.conv1(x)
        res = self.res_proj(x)

        x = self.body(x)
        x = x + res  # Residual connection

        # Pool along depth dimension
        x2d = torch.mean(x, dim=2)  # [B, C, H, W]

        # Apply attention modules
        x2d = self.eca(x2d)
        x2d = self.spatial_att(x2d)

        # Density map
        density = self.density_head(x2d)

        # Per-sample normalization
        max_val = density.amax(dim=(2, 3), keepdim=True)
        max_val = torch.where(max_val == 0, torch.ones_like(max_val), max_val)
        density = density / max_val

        # Regression output
        reg_value = self.reg_head(x2d)

        return density, reg_value


