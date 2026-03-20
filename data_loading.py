# -*- coding: utf-8 -*-
"""Data-loading utilities for the standalone fusion_selfsup project."""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

import data.rgbnir_stereo_pair as pair_mod
from data.rgbnir_stereo_sequence import RGBNIRStereoSequenceDataset


def _count_split_lines(path: str) -> int:
    n = 0
    with open(path, "r") as f:
        for ln in f:
            if ln.strip():
                n += 1
    return n


class LazyRGBNIRStereoPairDataset(torch.utils.data.Dataset):
    """
    Lightweight stereo-only RGB+NIR dataset for fusion_selfsup.

    Unlike the original eager pair dataset, this version does not stat every image in
    the split during __init__. It only reads the split file and resolves frame
    content on demand in __getitem__, which avoids very slow startup on network
    filesystems.
    """

    def __init__(
        self,
        root: str,
        size,
        augment: bool,
        frame_list_file: str,
        disp_root: Optional[str] = None,
        load_disp: bool = False,
        allow_inline_disp_fallback: bool = False,
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.root = root
        self.size = tuple(size)
        self.augment = bool(augment)
        self.disp_root = disp_root
        self.load_disp = bool(load_disp)
        self.allow_inline_disp_fallback = bool(allow_inline_disp_fallback)
        self.hflip_prob = float(hflip_prob)
        self.color_jitter = bool(color_jitter)
        self._calib_cache: Dict[str, Any] = {}

        with open(frame_list_file, "r") as f:
            self.frame_dirs = [
                pair_mod._resolve_frame_dir(self.root, ln.strip())
                for ln in f
                if ln.strip()
            ]
        if max_samples is not None:
            self.frame_dirs = self.frame_dirs[: int(max_samples)]
        if len(self.frame_dirs) == 0:
            raise RuntimeError(f"[LazyRGBNIRStereoPairDataset] no frame found in split: {frame_list_file}")

    def __len__(self) -> int:
        return len(self.frame_dirs)

    def _read_calib_cached(self, seq_root: str):
        cached = self._calib_cache.get(seq_root)
        if cached is None:
            cached = pair_mod._read_calib(seq_root)
            self._calib_cache[seq_root] = cached
        return cached

    def _resolve_paths(self, frame_dir: str):
        rgb_base = os.path.join(frame_dir, "rgb")
        nir_base = os.path.join(frame_dir, "nir")
        l_rgb = pair_mod._find_one_of(rgb_base, [f"left{e}" for e in pair_mod._IMAGE_EXTS])
        r_rgb = pair_mod._find_one_of(rgb_base, [f"right{e}" for e in pair_mod._IMAGE_EXTS])
        l_nir = pair_mod._find_one_of(nir_base, [f"left{e}" for e in pair_mod._IMAGE_EXTS])
        r_nir = pair_mod._find_one_of(nir_base, [f"right{e}" for e in pair_mod._IMAGE_EXTS])
        if not (l_rgb and r_rgb and l_nir and r_nir):
            raise FileNotFoundError(f"[LazyRGBNIRStereoPairDataset] incomplete frame assets: {frame_dir}")
        return l_rgb, r_rgb, l_nir, r_nir

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Retry a few nearby samples instead of failing the whole loader on a bad frame.
        for retry in range(min(8, len(self.frame_dirs))):
            cur_idx = (idx + retry) % len(self.frame_dirs)
            frame_dir = self.frame_dirs[cur_idx]
            seq_root = os.path.dirname(os.path.normpath(frame_dir))
            try:
                K_np, baseline = self._read_calib_cached(seq_root)
                left_rgb_path, right_rgb_path, left_nir_path, right_nir_path = self._resolve_paths(frame_dir)
                break
            except Exception:
                if retry == min(8, len(self.frame_dirs)) - 1:
                    raise
        K = torch.from_numpy(K_np.copy()).float()

        left_rgb = pair_mod._load_rgb(left_rgb_path)
        right_rgb = pair_mod._load_rgb(right_rgb_path)
        left_nir = pair_mod._load_nir(left_nir_path)
        right_nir = pair_mod._load_nir(right_nir_path)

        disp_L_np = None
        disp_R_np = None
        if self.load_disp:
            rel = None
            try:
                rel = os.path.relpath(os.path.abspath(frame_dir), os.path.abspath(self.root))
            except Exception:
                rel = None
            if self.disp_root and rel:
                disp_L_np = pair_mod._load_disp(os.path.join(self.disp_root, rel, "disp_left.npy"))
                disp_R_np = pair_mod._load_disp(os.path.join(self.disp_root, rel, "disp_right.npy"))
            if self.allow_inline_disp_fallback:
                if disp_L_np is None:
                    disp_L_np = pair_mod._load_disp(os.path.join(frame_dir, "disp_left.npy"))
                if disp_R_np is None:
                    disp_R_np = pair_mod._load_disp(os.path.join(frame_dir, "disp_right.npy"))

        if self.augment and self.color_jitter:
            params = pair_mod._color_jitter_params()
            left_rgb = pair_mod._apply_rgb_jitter(left_rgb, params)
            right_rgb = pair_mod._apply_rgb_jitter(right_rgb, params)

        H1, W1 = self.size
        W0, H0 = left_rgb.size
        sx, sy = W1 / W0, H1 / H0

        left_rgb = left_rgb.resize((W1, H1), pair_mod.Image.BILINEAR)
        right_rgb = right_rgb.resize((W1, H1), pair_mod.Image.BILINEAR)
        left_nir = left_nir.resize((W1, H1), pair_mod.Image.BILINEAR)
        right_nir = right_nir.resize((W1, H1), pair_mod.Image.BILINEAR)
        K = pair_mod._resize_K(K.unsqueeze(0), sx, sy).squeeze(0)

        def _rz_disp(d_np: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            if d_np is None:
                return None
            t = torch.from_numpy(d_np).unsqueeze(0).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(H1, W1), mode="bilinear", align_corners=False)
            t = t * sx
            return t.squeeze(0).squeeze(0)

        disp_L_t = _rz_disp(disp_L_np)
        disp_R_t = _rz_disp(disp_R_np)

        if self.augment and random.random() < self.hflip_prob:
            left_rgb, right_rgb = TF.hflip(right_rgb), TF.hflip(left_rgb)
            left_nir, right_nir = TF.hflip(right_nir), TF.hflip(left_nir)
            K = pair_mod._hflip_K(K.unsqueeze(0), width_after=W1).squeeze(0)
            disp_L_out = TF.hflip(disp_R_t.unsqueeze(0)).squeeze(0) if disp_R_t is not None else None
            disp_R_out = TF.hflip(disp_L_t.unsqueeze(0)).squeeze(0) if disp_L_t is not None else None
        else:
            disp_L_out = disp_L_t
            disp_R_out = disp_R_t

        sample = dict(
            L_t_rgbn=pair_mod._stack_rgbn(left_rgb, left_nir),
            R_t_rgbn=pair_mod._stack_rgbn(right_rgb, right_nir),
            K=K,
            fx=K[0, 0].view(1),
            baseline=torch.tensor([float(baseline)], dtype=torch.float32),
            meta=dict(seq_root=seq_root, frame_dir=frame_dir),
        )
        if disp_L_out is not None:
            sample["disp_igev_L"] = disp_L_out
            sample["disp_igev_t"] = disp_L_out
        if disp_R_out is not None:
            sample["disp_igev_R"] = disp_R_out
        return sample


def _build_dataset_from_list(cfg: Dict[str, Any], split: str) -> Optional[torch.utils.data.Dataset]:
    """Build a dataset strictly from the split list file."""
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    size = tuple(data_cfg["size"])
    mode = data_cfg.get("mode", "stereo_temporal")
    stride = int(data_cfg.get("temporal_stride", 1))
    augment = bool(data_cfg.get("augment", True)) if split == "train" else False
    disp_root = data_cfg.get("disp_root", None)
    load_disp = bool(data_cfg.get("load_disp", False))
    allow_inline_disp_fallback = bool(data_cfg.get("allow_inline_disp_fallback", False))
    max_samples = data_cfg.get("max_samples", None)

    list_key = f"{split}_list"
    list_path = data_cfg.get(list_key, None)

    if list_path is None or (not os.path.isfile(list_path)):
        return None

    if mode == "stereo_only":
        # For very small debug splits, keep the original eager dataset behavior.
        # It is closer to the old project and avoids per-sample filesystem checks.
        if max_samples is None:
            try:
                n_lines = _count_split_lines(list_path)
            except Exception:
                n_lines = 10**9
            if n_lines <= 128:
                return pair_mod.RGBNIRStereoPairDataset(
                    root=root,
                    size=size,
                    augment=augment,
                    frame_list_file=list_path,
                    disp_root=disp_root,
                    load_disp=load_disp,
                    allow_inline_disp_fallback=allow_inline_disp_fallback,
                )
        return LazyRGBNIRStereoPairDataset(
            root=root,
            size=size,
            augment=augment,
            frame_list_file=list_path,
            disp_root=disp_root,
            load_disp=load_disp,
            allow_inline_disp_fallback=allow_inline_disp_fallback,
            max_samples=max_samples,
        )
    if mode == "temporal_only":
        return RGBNIRStereoSequenceDataset(
            root=root,
            size=size,
            stride=stride,
            augment=augment,
            frame_list_file=list_path,
            disp_root=disp_root,
            load_disp=load_disp,
            allow_inline_disp_fallback=allow_inline_disp_fallback,
        )

    return RGBNIRStereoSequenceDataset(
        root=root,
        size=size,
        stride=stride,
        augment=augment,
        frame_list_file=list_path,
        disp_root=disp_root,
        load_disp=load_disp,
        allow_inline_disp_fallback=allow_inline_disp_fallback,
    )


def build_dataloader_from_list(cfg: Dict[str, Any], split: str) -> Optional[DataLoader]:
    ds = _build_dataset_from_list(cfg, split)
    if ds is None:
        return None
    bs = int(cfg["data"].get("batch_size", 4))
    nw = int(cfg["data"].get("num_workers", 4))
    shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        sampler=None,
        num_workers=nw,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch
