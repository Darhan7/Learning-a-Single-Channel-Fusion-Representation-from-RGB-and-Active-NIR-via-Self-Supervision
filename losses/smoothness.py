# -*- coding: utf-8 -*-
"""
Edge-aware smoothness regularization.
Commonly used in self-supervised depth/disparity learning.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def edge_aware_smooth(disp: torch.Tensor, img_for_edge: torch.Tensor) -> torch.Tensor:
    """
    disp: [B,1,H,W] disparity
    img_for_edge: [B,1 or 3,H,W] edge-guidance image
    (typically left RGB grayscale or NIR)
    """
    if img_for_edge.size(1) == 3:
        gray = 0.299 * img_for_edge[:, 0:1] + 0.587 * img_for_edge[:, 1:2] + 0.114 * img_for_edge[:, 2:3]
    else:
        gray = img_for_edge

    dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]
    dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]

    gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]

    weight_x = torch.exp(-torch.abs(gx))
    weight_y = torch.exp(-torch.abs(gy))

    sx = (torch.abs(dx) * weight_x).mean()
    sy = (torch.abs(dy) * weight_y).mean()
    return sx + sy
