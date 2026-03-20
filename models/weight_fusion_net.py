# -*- coding: utf-8 -*-
"""
Weight-based 1ch fusion:
- predict per-pixel weights w over (RGB-V, NIR)
- output F = w0 * V + w1 * NIR
- keep training in 0..1, export-ready output in 0..255 (float32)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion_attention_blocks import BasicEncoder, ResidualBlock, BAttentionFeatureFusion


def _rgb_to_value(rgb01: torch.Tensor) -> torch.Tensor:
    return torch.amax(rgb01, dim=1, keepdim=True)


def _guided_filter_1ch(guide: torch.Tensor, src: torch.Tensor, radius: int = 5, eps: float = 1e-5) -> torch.Tensor:
    if radius <= 0:
        return src
    k = 2 * radius + 1
    n = float(k * k)

    def box(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius) * n

    mean_i = box(guide) / n
    mean_p = box(src) / n
    corr_i = box(guide * guide) / n
    corr_ip = box(guide * src) / n
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = box(a) / n
    mean_b = box(b) / n
    return (mean_a * guide + mean_b).clamp(0.0, 1.0)


def _replace_bn_with_gn(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            ch = int(child.num_features)
            groups = min(32, ch)
            while groups > 1 and (ch % groups != 0):
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, ch, affine=True))
        else:
            _replace_bn_with_gn(child)


class AuxDispHead1Ch(nn.Module):
    def __init__(self, dmin_px: float = 0.05, dmax_px: float = 256.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Softplus(beta=1.0),
        )
        self.register_buffer("dmin_px", torch.tensor(float(dmin_px)), persistent=False)
        self.register_buffer("dmax_px", torch.tensor(float(dmax_px)), persistent=False)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        disp = self.net(x01)
        disp = torch.nan_to_num(disp, nan=0.0, posinf=self.dmax_px, neginf=0.0)
        return disp.clamp(self.dmin_px, self.dmax_px)


class WeightFusionNet(nn.Module):
    def __init__(
        self,
        feat_dim: int = 256,
        reduction: int = 4,
        use_guided_filter: bool = False,
        guided_radius: int = 5,
    ):
        super().__init__()
        self.encoder = BasicEncoder(downsample=2, output_dim=int(feat_dim))
        self.fusion = BAttentionFeatureFusion(in_channels=int(feat_dim), reduction=int(reduction))
        # Attention fusion blocks use BatchNorm(track_running_stats=False), which is unstable for bs=1 and 1x1 maps.
        _replace_bn_with_gn(self.fusion)
        mid = max(64, int(feat_dim // 2))
        low = max(32, int(mid // 2))
        self.weight_head = nn.Sequential(
            ResidualBlock(int(feat_dim), mid, norm_fn="batch"),
            ResidualBlock(mid, low, norm_fn="batch"),
            nn.Conv2d(low, 2, kernel_size=3, padding=1),
        )
        self.use_guided_filter = bool(use_guided_filter)
        self.guided_radius = int(guided_radius)

    def forward(self, rgb01: torch.Tensor, nir01: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb01 = rgb01.clamp(0.0, 1.0)
        nir01 = nir01.clamp(0.0, 1.0)
        v01 = _rgb_to_value(rgb01)

        rgb_in = rgb01 * 2.0 - 1.0
        nir_in = nir01.repeat(1, 3, 1, 1) * 2.0 - 1.0
        rgb_feat = self.encoder(rgb_in)
        nir_feat = self.encoder(nir_in)
        fused_feat = self.fusion(rgb_feat, nir_feat)

        logits = self.weight_head(fused_feat)
        logits = F.interpolate(logits, size=rgb01.shape[-2:], mode="bilinear", align_corners=False)
        weights = torch.softmax(logits, dim=1)
        w_v = weights[:, 0:1]
        w_n = weights[:, 1:2]

        f01 = (w_v * v01 + w_n * nir01).clamp(0.0, 1.0)
        if self.use_guided_filter:
            f01 = _guided_filter_1ch(nir01, f01, radius=self.guided_radius)
        f255 = (f01 * 255.0).clamp(0.0, 255.0).to(torch.float32)
        return {
            "f255": f255,
            "f01": (f255 / 255.0),
            "weights": weights,
            "w_v": w_v,
            "w_n": w_n,
            "v01": v01,
        }
