"""Shared CIFAR-style ResNet backbone with a pluggable attention slot.

The same BasicBlock/ResNet code is used for every arm of the audit (None,
SE, BAM, CBAM); only `attention_type` changes. The attention module is
inserted after the second BatchNorm and before the residual addition, and
the identity shortcut never passes through it.
"""

import torch
import torch.nn as nn

from attention import SEBlock, BAMBlock, CBAMBlock


def get_attention_module(attention_type, channels, reduction=16):
    if attention_type is None or attention_type == "none":
        return None
    if attention_type == "SE":
        return SEBlock(channels, reduction=reduction)
    if attention_type == "BAM":
        return BAMBlock(channels, reduction_ratio=reduction)
    if attention_type == "CBAM":
        return CBAMBlock(channels, reduction_ratio=reduction)
    raise ValueError(f"Unknown attention_type: {attention_type!r}")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, attention_type=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.attention = get_attention_module(attention_type, out_channels, reduction=reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.attention is not None:
            out = self.attention(out)

        out = self.relu(out + identity)
        return out


class ResNetCIFAR(nn.Module):
    """Small ResNet for 32x32 inputs (CIFAR-10), attention-type is a single
    argument so the backbone is identical across arms of the audit."""

    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, attention_type=None, reduction=16):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, attention_type=attention_type, reduction=reduction)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, attention_type=attention_type, reduction=reduction)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, attention_type=attention_type, reduction=reduction)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, attention_type=attention_type, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride, attention_type, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                BasicBlock(self.in_channels, out_channels, stride=s, attention_type=attention_type, reduction=reduction)
            )
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet_cifar(attention_type=None, num_classes=10, reduction=16):
    """4-stage, 2-blocks-per-stage ResNet (ResNet-18-style) for CIFAR-10."""
    return ResNetCIFAR(num_blocks=(2, 2, 2, 2), num_classes=num_classes, attention_type=attention_type, reduction=reduction)
