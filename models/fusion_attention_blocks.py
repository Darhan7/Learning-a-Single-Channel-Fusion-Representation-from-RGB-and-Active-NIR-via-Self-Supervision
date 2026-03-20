# -*- coding: utf-8 -*-
"""
Local copy of the minimal attention/backbone blocks used by fusion_selfsup.

This vendors only the parts that the root fusion pipeline actually depends on:
- ResidualBlock
- BasicEncoder
- BAttentionFeatureFusion and its attention helpers

The goal is to remove the runtime dependency on external fusion code while
preserving the same functional architecture for the main method.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, norm_fn: str = "group", stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = max(1, planes // 8)

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()
        else:
            raise ValueError(f"Unsupported norm_fn: {norm_fn}")

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 128,
        norm_fn: str = "batch",
        input_dim: int = 3,
        dropout: float = 0.0,
        downsample: int = 3,
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample
        self.input_dim = input_dim

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()
        else:
            raise ValueError(f"Unsupported norm_fn: {norm_fn}")

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x, dual_inp: bool = False):
        del dual_inp
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)
        return x


class LocalAttentionModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.local_conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.local_bn1 = nn.BatchNorm2d(hidden, track_running_stats=False)
        self.local_relu = nn.ReLU(inplace=False)
        self.local_conv2 = nn.Conv2d(hidden, in_channels, kernel_size=1)
        self.local_bn2 = nn.BatchNorm2d(in_channels, track_running_stats=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_conv1(x)
        x = self.local_bn1(x)
        x = self.local_relu(x)
        x = self.local_conv2(x)
        x = self.local_bn2(x)
        return x


class GlobalAttentionModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.global_bn1 = nn.BatchNorm2d(hidden, track_running_stats=False, eps=0.001)
        self.global_relu = nn.ReLU(inplace=False)
        self.global_conv2 = nn.Conv2d(hidden, in_channels, kernel_size=1)
        self.global_bn2 = nn.BatchNorm2d(in_channels, track_running_stats=False, eps=0.001)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.data *= 0.1
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_avg_pool(x).float()
        x = self.global_conv1(x)
        x = self.global_bn1(x)
        x = self.global_relu(x)
        x = self.global_conv2(x)
        x = self.global_bn2(x)
        return x


class MultiScaleChannelAttentionModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.local_attention = LocalAttentionModule(in_channels, reduction)
        self.global_attention = GlobalAttentionModule(in_channels, reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.local_attention(x) + self.global_attention(x)
        return self.sigmoid(out)


class BAttentionFeatureFusion(nn.Module):
    def __init__(self, in_channels: int = 128, reduction: int = 4):
        super().__init__()
        self.attention_rgb = MultiScaleChannelAttentionModule(in_channels, reduction)
        self.attention_nir = MultiScaleChannelAttentionModule(in_channels, reduction)
        self.attention_fusion = MultiScaleChannelAttentionModule(in_channels, reduction)

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor, debug_attention: bool = False):
        rgb_att = self.attention_rgb(rgb)
        nir_att = self.attention_nir(nir)

        sum_att = rgb_att + nir_att + 1e-6
        rgb_att = rgb * rgb_att / sum_att * 2
        nir_att = nir * nir_att / sum_att * 2
        att_features = rgb_att + nir_att

        att_fusion = self.attention_fusion(att_features)
        out = att_fusion * rgb + (1 - att_fusion) * nir
        if debug_attention:
            return att_fusion, rgb, nir
        return out
