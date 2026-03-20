# -*- coding: utf-8 -*-
"""
Intrinsic/disparity adaptation utilities synchronized with data augmentation:
- crop -> resize -> pad -> horizontal flip
- disparity scaling and sign flip for horizontal motion
"""
from __future__ import annotations
from typing import List, Tuple
import torch


def apply_crop_to_K(K: torch.Tensor, u0: float, v0: float) -> torch.Tensor:
    """
    Apply crop first. The principal point must be shifted by the crop offset.
    K: [B,3,3] or [3,3]
    """
    K = K.clone()
    K[..., 0, 2] -= float(u0)
    K[..., 1, 2] -= float(v0)
    return K


def apply_resize_to_K(K: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """
    Support both isotropic and anisotropic resize. Scale fx, fy, cx, cy independently.
    """
    K = K.clone()
    K[..., 0, 0] *= float(sx)
    K[..., 1, 1] *= float(sy)
    K[..., 0, 2] *= float(sx)
    K[..., 1, 2] *= float(sy)
    return K


def apply_pad_to_K(K: torch.Tensor, pad_left: int, pad_top: int) -> torch.Tensor:
    """
    Padding only shifts the principal point.
    """
    K = K.clone()
    K[..., 0, 2] += int(pad_left)
    K[..., 1, 2] += int(pad_top)
    return K


def apply_hflip_to_K(K: torch.Tensor, width_after: int) -> torch.Tensor:
    """
    After horizontal flip, reflect the principal point about the image center:
    cx' = (W - 1) - cx
    """
    K = K.clone()
    cx = K[..., 0, 2]
    K[..., 0, 2] = (int(width_after) - 1) - cx
    return K


def rescale_disparity(disp: torch.Tensor, sx: float, hflip: bool = False) -> torch.Tensor:
    """
    Disparity is a horizontal pixel shift: scale by sx, negate after horizontal flip.
    disp: [B,1,H,W]
    """
    d = disp * float(sx)
    if hflip:
        d = -d
    return d


def adapt_K_and_disp(
    K: torch.Tensor,
    ops: List[Tuple],
    disp: torch.Tensor | None = None,
    width_after: int | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compose multiple geometry updates in the given order.
    Supported ops:
      ("crop", u0, v0)
      ("resize", sx, sy)
      ("pad", left, top)
      ("hflip",)
    Return: (K_new, disp_new)
    """
    Kp = K
    dp = disp
    for op in ops:
        tag = op[0]
        if tag == "crop":
            _, u0, v0 = op
            Kp = apply_crop_to_K(Kp, u0, v0)
        elif tag == "resize":
            _, sx, sy = op
            Kp = apply_resize_to_K(Kp, sx, sy)
            if dp is not None:
                dp = rescale_disparity(dp, sx=sx, hflip=False)
        elif tag == "pad":
            _, left, top = op
            Kp = apply_pad_to_K(Kp, left, top)
        elif tag == "hflip":
            assert width_after is not None, "hflip requires width_after to be specified"
            Kp = apply_hflip_to_K(Kp, width_after)
            if dp is not None:
                dp = rescale_disparity(dp, sx=1.0, hflip=True)
        else:
            raise ValueError(f"Unknown op: {tag}")
    return Kp, dp
