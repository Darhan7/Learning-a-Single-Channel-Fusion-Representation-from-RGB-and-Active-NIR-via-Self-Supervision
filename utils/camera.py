# -*- coding: utf-8 -*-
"""
Geometry / camera utilities:
- depth_from_disp: convert pixel disparity to depth
- pixel2cam / cam2pixel: pixel <-> camera coordinate transforms
- normalize_grid: convert pixel coordinates to grid_sample coordinates
"""
from __future__ import annotations
from typing import Tuple
import torch


def _ensure_batch_K(K: torch.Tensor, B: int) -> torch.Tensor:
    """
    Make sure K has shape [B,3,3]. If [3,3] is given, expand without copying.
    """
    if K.dim() == 2:
        K = K.unsqueeze(0)
    assert K.dim() == 3 and K.size(1) == 3 and K.size(2) == 3
    if K.size(0) == 1 and B > 1:
        K = K.expand(B, -1, -1)
    return K


def depth_from_disp(disp_px: torch.Tensor,
                    fx: torch.Tensor,
                    baseline_m: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """
    disp_px: [B,1,H,W] pixel disparity (>0)
    fx: [B] or [B,1] or [B,1,1,1] (or scalar tensor), focal length in pixels
    baseline_m: [B] or [B,1] or [B,1,1,1] (or scalar tensor), baseline in meters
    return: depth_m [B,1,H,W]
    """
    # Broadcast to [B,1,1,1].
    while fx.dim() < 4: fx = fx.unsqueeze(-1)
    while baseline_m.dim() < 4: baseline_m = baseline_m.unsqueeze(-1)
    # Use absolute values to ignore baseline sign conventions.
    depth = fx.abs() * baseline_m.abs() / (disp_px.clamp_min(eps))
    return depth


def pixel2cam(uv1: torch.Tensor, depth: torch.Tensor, Kinv: torch.Tensor) -> torch.Tensor:
    """
    Back-project homogeneous pixel coordinates uv1 with depth into camera space.
    uv1: [B,3,H,W] as (x,y,1)
    depth: [B,1,H,W]
    Kinv: [B,3,3]
    return X_cam: [B,3,H,W]
    """
    B, _, H, W = uv1.shape
    Kinv = _ensure_batch_K(Kinv, B)
    uv1_flat = uv1.view(B, 3, -1)                    # [B,3,HW]
    X_cam = torch.bmm(Kinv, uv1_flat).view(B, 3, H, W)
    X_cam = X_cam * depth
    return X_cam


def cam2pixel(X_cam: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project camera coordinates onto the image plane.
    X_cam: [B,3,H,W]
    K: [B,3,3] or [3,3]
    return: (u,v,Z) -> [B,1,H,W]
    """
    B, _, H, W = X_cam.shape
    K = _ensure_batch_K(K, B)

    fx = K[:, 0, 0].view(B, 1, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1, 1)

    X = X_cam[:, 0:1]
    Y = X_cam[:, 1:2]
    Z = X_cam[:, 2:3].clamp_min(1e-3)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return u, v, Z


def make_base_uv1(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """
    Build a homogeneous pixel grid uv1: [B,3,H,W] storing (x,y,1).
    """
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    ones = torch.ones_like(x)
    uv1 = torch.stack([x, y, ones], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    return uv1


def normalize_grid(u: torch.Tensor, v: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert pixel coordinates (u,v) to F.grid_sample coordinates [B,H,W,2]
    using the align_corners=True convention:
        x_norm = 2*u/(W-1) - 1
        y_norm = 2*v/(H-1) - 1
    """
    x = 2.0 * (u / (W - 1)) - 1.0
    y = 2.0 * (v / (H - 1)) - 1.0
    grid = torch.stack([x, y], dim=-1)  # Resulting shape depends on caller layout.
    return grid
