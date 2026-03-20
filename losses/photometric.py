# -*- coding: utf-8 -*-
"""
Photometric utilities:
- SSIM + L1 combination
- RGB/NIR adaptive gate g(x, y)
- min-reprojection and auto-masking
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _avg_pool3(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


def ssim_map(x: torch.Tensor, y: torch.Tensor, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> torch.Tensor:
    """
    Simplified SSIM. Compute SSIM per channel, average across channels,
    and return the (1 - SSIM) / 2 map with shape [B,1,H,W].
    This follows the common Monodepth2-style implementation.
    """
    mu_x = _avg_pool3(x)
    mu_y = _avg_pool3(y)
    sigma_x = _avg_pool3(x * x) - mu_x * mu_x
    sigma_y = _avg_pool3(y * y) - mu_y * mu_y
    sigma_xy = _avg_pool3(x * y) - mu_x * mu_y

    ssim_num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim = torch.clamp((1 - ssim_num / (ssim_den + 1e-6)) / 2, 0, 1)  # [B,C,H,W]
    return ssim.mean(dim=1, keepdim=True)  # [B,1,H,W]


def ssim_l1(a: torch.Tensor, b: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    """
    photometric = alpha*SSIM + (1-alpha)*L1
    Inputs a and b can have 1 or 3 channels. Output shape is [B,1,H,W].
    """
    ssim = ssim_map(a, b)
    l1 = (a - b).abs().mean(dim=1, keepdim=True)
    return alpha * ssim + (1 - alpha) * l1


def rgb_nir_gate(IL_rgbn: torch.Tensor) -> torch.Tensor:
    """
    Hybrid global + local gate:
      - Global term: compare mean scene brightness (RGB grayscale vs NIR)
      - Local term: refine using edge strength and saturation/darkness differences
    Return g in [0,1], where g=1 favors NIR and g=0 favors RGB.
    """
    rgb = IL_rgbn[:, :3]
    nir = IL_rgbn[:, 3:4]
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

    mu_rgb = gray.mean(dim=(2, 3), keepdim=True)
    mu_nir = nir.mean(dim=(2, 3), keepdim=True)
    # Saturated / dark-region ratios.
    sat_rgb = (rgb > 0.98).float().mean(dim=1, keepdim=True)  # [B,1,H,W] after mean-> [B,1,1,1]
    sat_nir = (nir > 0.98).float()
    dark_rgb = (gray < 0.05).float()
    dark_nir = (nir < 0.05).float()

    def grad_mag(x):
        gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (1, 0, 0, 0))
        gy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 1, 0))
        return (gx * gx + gy * gy).sqrt()

    edge_rgb = grad_mag(gray)
    edge_nir = grad_mag(nir)

    g_global = torch.sigmoid(5.0 * (mu_nir - mu_rgb))                 # [B,1,1,1]
    g_local = 2.5 * (edge_nir - edge_rgb) - 3.0 * (sat_nir - sat_rgb) - 2.0 * (dark_nir - dark_rgb)
    g = (g_global + g_local).clamp(0.0, 1.0)
    return g


def min_reprojection(losses: list[torch.Tensor]) -> torch.Tensor:
    """
    Take the per-pixel minimum across a set of reprojection losses
    (for example right view, t-1, t+1, ...).
    Each entry must have shape [B,1,H,W].
    """
    assert len(losses) >= 1
    stacked = torch.stack(losses, dim=0)  # [S,B,1,H,W]
    return torch.min(stacked, dim=0).values  # [B,1,H,W]


def auto_masking(id_loss: torch.Tensor, reproj_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Auto-masking: ignore pixels that look more similar to the source image
    before warping (for example static regions or occlusions).
    Return (masked_loss, mask).
    """
    with torch.no_grad():
        mask = (reproj_loss < id_loss).float()  # [B,1,H,W]
    num = (reproj_loss * mask).sum()
    den = mask.sum() + 1e-6
    masked = num / den
    return masked, mask
