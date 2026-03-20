from __future__ import annotations
from typing import List, Optional
import os
import torch
import torchvision.utils as vutils

def normalize_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _normalize_fixed(x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    z = (x - vmin) / (vmax - vmin + 1e-6)
    return z.clamp(0, 1)

def disp_to_rgb(disp: torch.Tensor, vmin: Optional[float] = None, vmax: Optional[float] = None) -> torch.Tensor:
    """
    Convert disparity [B,1,H,W] to a simple pseudo-color tensor [B,3,H,W].
    - If vmin/vmax are provided, use a fixed normalization range.
    - Otherwise fall back to per-frame min/max normalization.
    Pseudo-color mapping: r = z, g = sqrt(z), b = 1 - z.
    """
    assert disp.dim() == 4 and disp.size(1) == 1
    if vmin is None or vmax is None:
        z = normalize_01(disp)
    else:
        z = _normalize_fixed(disp, vmin, vmax)
    r = z
    g = torch.clamp(z.sqrt(), 0, 1)
    b = torch.clamp(1.0 - z, 0, 1)
    return torch.cat([r, g, b], dim=1)

def save_panel(paths: List[str], tensors: List[torch.Tensor], nrow: int = 4) -> None:
    os.makedirs(os.path.dirname(paths[0]) if paths else ".", exist_ok=True)
    for p, t in zip(paths, tensors):
        vutils.save_image(t, p, nrow=nrow, normalize=False)
