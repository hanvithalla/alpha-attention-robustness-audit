"""SE, BAM, and CBAM attention modules.

Each module maps a feature tensor (B, C, H, W) to a recalibrated tensor of the
same shape, so they are interchangeable at a single insertion point in the
shared backbone (see model.py).
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018): channel-only attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelGate(nn.Module):
    """Shared avg+max pooled channel gate used by BAM and CBAM."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        return avg_out + max_out  # channel logits, (b, c)


class SpatialGateCBAM(nn.Module):
    """CBAM spatial gate: channel-pooled 2-plane map -> conv -> sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(y))  # (b, 1, h, w)


class SpatialGateBAM(nn.Module):
    """BAM spatial gate: 1x1 channel reduction then dilated 3x3 convs."""

    def __init__(self, channels, reduction=16, dilation=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)  # channel logits over space, (b, 1, h, w)


class CBAMBlock(nn.Module):
    """Sequential channel-then-spatial attention (Woo et al., 2018)."""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction=reduction)
        self.spatial_gate = SpatialGateCBAM(kernel_size=spatial_kernel)

    def forward(self, x):
        b, c, _, _ = x.shape
        x = x * torch.sigmoid(self.channel_gate(x)).view(b, c, 1, 1)
        x = x * self.spatial_gate(x)
        return x


class BAMBlock(nn.Module):
    """Parallel channel + spatial attention combined additively (Park et al., 2018)."""

    def __init__(self, channels, reduction=16, dilation=4):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction=reduction)
        self.spatial_gate = SpatialGateBAM(channels, reduction=reduction, dilation=dilation)

    def forward(self, x):
        b, c, _, _ = x.shape
        ch_att = self.channel_gate(x).view(b, c, 1, 1)  # (b, c, 1, 1)
        sp_att = self.spatial_gate(x)                     # (b, 1, h, w)
        att = torch.sigmoid(ch_att + sp_att)               # broadcasts to (b, c, h, w)
        return x + x * att
