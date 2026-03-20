# -*- coding: utf-8 -*-
"""
Generic warping utilities:
- warp_stereo_rectified: warp the right view into the left view using left disparity
- warp_temporal_projective: project a neighboring frame into the current frame
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

from .camera import make_base_uv1, pixel2cam, cam2pixel, normalize_grid


@torch.no_grad()
def _make_norm_grid(B: int, H: int, W: int, device, dtype):
    """
    Build a normalized sampling grid with align_corners=True.
    """
    y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.unsqueeze(0).expand(B, -1, -1)  # [B,H,W]
    yy = yy.unsqueeze(0).expand(B, -1, -1)
    return xx, yy


def warp_stereo_rectified(imgR: torch.Tensor,
                          dispL: torch.Tensor,
                          padding_mode: str = "border") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Warp the right image into the left-image coordinate frame using left disparity.
    imgR:  [B,C,H,W]
    dispL: [B,1,H,W]
    return: (warped, valid_mask, grid)
    """
    B, C, H, W = imgR.shape
    xx, yy = _make_norm_grid(B, H, W, imgR.device, imgR.dtype)
    disp_norm = 2.0 * dispL.squeeze(1) / (W - 1)  # Pixel shift -> normalized shift.
    grid = torch.stack([xx - disp_norm, yy], dim=-1)  # [B,H,W,2]

    warped = F.grid_sample(imgR, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    valid_x = (grid[..., 0] > -1.0) & (grid[..., 0] < 1.0)
    valid_y = (grid[..., 1] > -1.0) & (grid[..., 1] < 1.0)
    valid = (valid_x & valid_y).unsqueeze(1).float()
    return warped, valid, grid


def warp_temporal_projective(src_img: torch.Tensor,
                             depth_t: torch.Tensor,
                             K: torch.Tensor,
                             T_t2s: torch.Tensor,
                             padding_mode: str = "border") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Warp source frame s into target frame t:
    src_img: [B,C,H,W] at time s
    depth_t: [B,1,H,W] depth / relative depth at time t
    K:       [B,3,3] or [3,3]
    T_t2s:   [B,4,4], PoseNet-estimated relative pose from t to s
    return: (Iw, valid_mask, grid)
    """
    B, C, H, W = src_img.shape
    device, dtype = src_img.device, src_img.dtype

    uv1 = make_base_uv1(B, H, W, device, dtype)                   # [B,3,H,W]
    Kinv = torch.inverse(K if K.dim() == 3 else K.unsqueeze(0))  # [B,3,3]

    # Target pixels + depth -> target camera coordinates.
    Xt = pixel2cam(uv1, depth_t, Kinv)                            # [B,3,H,W]

    # Transform into the source camera frame.
    R = T_t2s[:, :3, :3]
    t = T_t2s[:, :3,  3].view(B, 3, 1, 1)
    Xs = torch.einsum('bij,bjhw->bihw', R, Xt) + t

    # Project to source-frame pixel coordinates.
    u, v, Zs = cam2pixel(Xs, K)                                   # [B,1,H,W]
    grid = normalize_grid(u, v, H, W).squeeze(1)                  # [B,H,W,2]

    Iw = F.grid_sample(src_img, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    valid_x = (grid[..., 0] > -1.0) & (grid[..., 0] < 1.0)
    valid_y = (grid[..., 1] > -1.0) & (grid[..., 1] < 1.0)
    valid_z = (Zs > 1e-3).squeeze(1)
    valid = (valid_x & valid_y & valid_z).unsqueeze(1).float()
    return Iw, valid, grid
