"""SE, BAM, and CBAM attention modules.

Each module maps a feature tensor (B, C, H, W) to a recalibrated tensor of the
same shape, so they are interchangeable at a single insertion point in the
shared backbone (see model.py).

Provenance (see docs/PROVENANCE.md for the full record)
-------------------------------------------------------
SE    reimplemented from Eq. 2-3 of Hu et al., "Squeeze-and-Excitation
      Networks" (arXiv:1709.01507). This is the one reimplementation the
      study protocol allows: the official repo (hujie-frank/SENet,
      Apache-2.0) is Caffe-only (.prototxt/.cpp/.cu) and has no Python.
BAM   ported from MODELS/bam.py, Jongchan/attention-module (MIT).
CBAM  ported from MODELS/cbam.py, Jongchan/attention-module (MIT).

BAM and CBAM keep the official module structure, including the details that
distinguish them: BAM's channel gate is average-pool-only with BatchNorm1d
inside the MLP and fuses its two gates *multiplicatively*, while CBAM's is an
avg+max shared MLP applied *sequentially* before a BN-carrying spatial conv.
Those differences are the channel-vs-spatial contrast this study measures, so
the two must not share gate code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Flatten(nn.Module):
    """MODELS/{bam,cbam}.py::Flatten."""

    def forward(self, x):
        return x.view(x.size(0), -1)


# --------------------------------------------------------------------------
# SE -- reimplemented from the paper (the one allowed reimplementation)
# --------------------------------------------------------------------------


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018): channel-only attention.

    Eq. 2 (squeeze): global average pool over HxW -> z in R^C.
    Eq. 3 (excitation): s = sigma(W2 delta(W1 z)), W1 in R^{C/r x C}.
    Scale: channel-wise rescale of the input by s.

    Reduction ratio r is a recorded design-space point, not a re-derivation
    of the paper's tuned value; r=16 matches the paper's default.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = channels // reduction
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


# --------------------------------------------------------------------------
# BAM -- ported from MODELS/bam.py (Jongchan/attention-module, MIT)
# --------------------------------------------------------------------------


class BAMChannelGate(nn.Module):
    """MODELS/bam.py::ChannelGate.

    Average pooling ONLY (no max branch), and BatchNorm1d between the MLP
    layers. This is what distinguishes BAM's channel gate from CBAM's.
    """

    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super().__init__()
        dims = [gate_channel] + [gate_channel // reduction_ratio] * num_layers + [gate_channel]
        layers = [_Flatten()]
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.gate_c = nn.Sequential(*layers)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)


class BAMSpatialGate(nn.Module):
    """MODELS/bam.py::SpatialGate -- 1x1 reduce, dilated 3x3 stack, 1x1 to 1."""

    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super().__init__()
        hidden = gate_channel // reduction_ratio
        layers = [
            nn.Conv2d(gate_channel, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        ]
        for _ in range(dilation_conv_num):
            layers += [
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=dilation_val, dilation=dilation_val),
                nn.BatchNorm2d(hidden),
                nn.ReLU(),
            ]
        layers.append(nn.Conv2d(hidden, 1, kernel_size=1))
        self.gate_s = nn.Sequential(*layers)

    def forward(self, x):
        return self.gate_s(x).expand_as(x)


class BAMBlock(nn.Module):
    """MODELS/bam.py::BAM -- parallel channel + spatial gates.

    NOTE (paper vs. repo): Park et al. (arXiv:1807.06514) specify additive
    fusion, sigma(Mc + Ms). The official repo multiplies the two gate maps
    instead: att = 1 + sigmoid(Mc * Ms). Milestone 4 requires the repo's
    definition, so the multiplicative form is used here and the discrepancy
    is reported in the write-up rather than silently resolved.
    """

    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super().__init__()
        self.channel_att = BAMChannelGate(gate_channel, reduction_ratio)
        self.spatial_att = BAMSpatialGate(gate_channel, reduction_ratio, dilation_conv_num, dilation_val)

    def forward(self, x):
        att = 1 + torch.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att * x


# --------------------------------------------------------------------------
# CBAM -- ported from MODELS/cbam.py (Jongchan/attention-module, MIT)
# --------------------------------------------------------------------------


class BasicConv(nn.Module):
    """MODELS/cbam.py::BasicConv -- conv (+BN) (+ReLU)."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CBAMChannelGate(nn.Module):
    """MODELS/cbam.py::ChannelGate -- avg AND max through a shared MLP."""

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max")):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            _Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = tuple(pool_types)

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                pooled = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == "max":
                pooled = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type!r}")
            channel_att_raw = self.mlp(pooled)
            channel_att_sum = channel_att_raw if channel_att_sum is None else channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    """MODELS/cbam.py::ChannelPool -- concat(max, mean) over the channel axis."""

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class CBAMSpatialGate(nn.Module):
    """MODELS/cbam.py::SpatialGate -- ChannelPool -> 7x7 BasicConv -> sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_out = self.spatial(self.compress(x))
        return x * torch.sigmoid(x_out)


class CBAMBlock(nn.Module):
    """MODELS/cbam.py::CBAM -- sequential channel-then-spatial attention."""

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max"), no_spatial=False):
        super().__init__()
        self.ChannelGate = CBAMChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = CBAMSpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
