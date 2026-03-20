# -*- coding: utf-8 -*-
"""
PoseNet (2ch input) based on ResNet18.
Input: concat(I_fuse_t, I_fuse_s) -> [B,2,H,W]
Output: 6DoF pose vector (axis-angle + translation)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class PoseNet2Ch(nn.Module):
    def __init__(self):
        super().__init__()
        base = tvm.resnet18(weights=None)
        # Replace the first conv layer to accept a 2-channel fused-image pair.
        base.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Keep only the encoder trunk and drop the classification head.
        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    @staticmethod
    def to_matrix(pose_vec: torch.Tensor) -> torch.Tensor:
        B = pose_vec.shape[0]
        rot = pose_vec[:, :3]
        trans = pose_vec[:, 3:].unsqueeze(-1)
        angle = torch.norm(rot, dim=1, keepdim=True) + 1e-8
        axis = rot / angle
        angle = angle.view(-1, 1, 1)

        def skew(a: torch.Tensor) -> torch.Tensor:
            x, y, z = a[:, 0], a[:, 1], a[:, 2]
            O = torch.zeros_like(x)
            return torch.stack([
                torch.stack([O, -z, y], dim=1),
                torch.stack([z, O, -x], dim=1),
                torch.stack([-y, x, O], dim=1),
            ], dim=1)

        K = skew(axis)
        I = torch.eye(3, device=pose_vec.device, dtype=pose_vec.dtype).unsqueeze(0).expand(B, -1, -1)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

        T = torch.zeros(B, 4, 4, device=pose_vec.device, dtype=pose_vec.dtype)
        T[:, :3, :3] = R
        T[:, :3, 3:4] = trans
        T[:, 3, 3] = 1.0
        return T
