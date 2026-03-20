# -*- coding: utf-8 -*-
"""
RGBNIRStereoSequenceDataset: stereo + temporal dataset with (t-1, t, t+1)
using raw RGB+NIR inputs.
- Requires frame_list_file and builds samples in per-sequence temporal order.
- Keeps only frames with a complete (t-1, t, t+1) triplet (default stride=1).
- Output:
  {
    L_t_rgbn, R_t_rgbn,
    L_tm1_rgbn, L_tp1_rgbn,   # Only neighboring left frames are used for temporal reprojection.
    K, fx, baseline,
    meta: { seq_root, frame_dir, frame_dir_tm1, frame_dir_tp1 }
  }
"""
from __future__ import annotations
import os
import glob
import random
from typing import Tuple, Dict, Any, List, Optional, DefaultDict
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from utils.intrinsics_adapter import apply_resize_to_K as _resize_K
from utils.intrinsics_adapter import apply_hflip_to_K as _hflip_K

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _looks_like_frame_dir(path: str) -> bool:
    """Lightweight check for a frame directory containing rgb/ and nir/ subfolders."""
    rgb_dir = os.path.join(path, "rgb")
    nir_dir = os.path.join(path, "nir")
    return os.path.isdir(path) and os.path.isdir(rgb_dir) and os.path.isdir(nir_dir)


def _expand_to_frame_dirs(path: str) -> List[str]:
    """
    If path already points to a frame directory, return it directly.
    Otherwise scan one level below and collect child directories that look like frames.
    This allows list files to contain sequence roots as well.
    """
    if _looks_like_frame_dir(path):
        return [path]

    if not os.path.isdir(path):
        return []

    frames: List[str] = []
    try:
        subdirs = sorted(os.listdir(path))
    except OSError:
        return []

    for name in subdirs:
        sub = os.path.join(path, name)
        if _looks_like_frame_dir(sub):
            frames.append(sub)
    return frames


def _normdir(p: str) -> str:
    return os.path.normpath(p.rstrip("/"))

def _resolve_frame_dir(root: str, raw: str) -> str:
    path = os.path.expanduser(os.path.expandvars(raw))
    if root and not os.path.isabs(path):
        path = os.path.join(root, path)
    return _normdir(path)

def _find_one_of(base: str, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.isfile(p):
            return p
    return None

def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def _load_nir(path: str) -> Image.Image:
    return Image.open(path).convert("L")

def _stack_rgbn(rgb: Image.Image, nir: Image.Image) -> torch.Tensor:
    t_rgb = TF.to_tensor(rgb)  # [3,H,W], 0..1
    t_nir = TF.to_tensor(nir)  # [1,H,W], 0..1
    return torch.cat([t_rgb, t_nir], dim=0)

def _read_calib(seq_root: str) -> Tuple[np.ndarray, float]:
    rect_npz = os.path.join(seq_root, "rect", "calib_rectified.npz")
    calib_npz = os.path.join(seq_root, "calibration.npz")

    K = None
    baseline = None

    if os.path.isfile(rect_npz):
        d = np.load(rect_npz, allow_pickle=True)
        if "K_rect_left" in d:
            K = d["K_rect_left"].astype(np.float64)
        if "P2" in d:
            P2 = d["P2"].astype(np.float64)
            fx = P2[0, 0]
            Tx = P2[0, 3] / (fx + 1e-12)
            baseline = abs(Tx)
    if (K is None or baseline is None) and os.path.isfile(calib_npz):
        d = np.load(calib_npz, allow_pickle=True)
        if K is None and "mtx_left" in d:
            K = d["mtx_left"].astype(np.float64)
        if baseline is None:
            if "P2" in d:
                P2 = d["P2"].astype(np.float64)
                fx = P2[0, 0]
                Tx = P2[0, 3] / (fx + 1e-12)
                baseline = abs(Tx)
            elif "T" in d:
                T = d["T"].astype(np.float64).reshape(3, 1)
                baseline = float(np.linalg.norm(T))

    if K is None or baseline is None:
        raise FileNotFoundError(f"Failed to parse K or baseline from {rect_npz} or {calib_npz}.")

    # In this dataset the baseline is often stored in millimeters (~100-300).
    # Convert it to meters.
    if float(baseline) > 10.0:
        baseline = float(baseline) * 1e-3

    return K, float(baseline)

def _color_jitter_params():
    b = 0.2 * (random.random() - 0.5) * 2 + 1.0
    c = 0.2 * (random.random() - 0.5) * 2 + 1.0
    s = 0.2 * (random.random() - 0.5) * 2 + 1.0
    h = 0.02 * (random.random() - 0.5) * 2
    return b, c, s, h

def _apply_rgb_jitter(img: Image.Image, params) -> Image.Image:
    b, c, s, h = params
    img = TF.adjust_brightness(img, b)
    img = TF.adjust_contrast(img, c)
    img = TF.adjust_saturation(img, s)
    img = TF.adjust_hue(img, h)
    return img

def _load_disp(path: str) -> Optional[np.ndarray]:
    if os.path.isfile(path):
        try:
            disp = np.load(path).astype(np.float32)
            # 1) Replace NaN / Inf with zeros and treat them as invalid pixels.
            disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
            # 2) Clamp to a reasonable disparity range.
            max_disp = 256.0
            disp = np.clip(disp, 0.0, max_disp)
            return disp
        except:
            return None
    return None

class RGBNIRStereoSequenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        size: Tuple[int, int] = (384, 512),  # (H, W)
        stride: int = 1,
        augment: bool = True,
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
        frame_list_file: Optional[str] = None,  # Required: one frame directory per line.
        disp_root: Optional[str] = None,        # Optional disparity-cache root.
        load_disp: bool = False,                # Strict self-supervision keeps this disabled.
        allow_inline_disp_fallback: bool = False,  # Fall back to frame_dir/disp_*.npy if cache is missing.
    ):
        super().__init__()
        self.root = root
        self.size = size
        self.stride = int(stride)
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.disp_root = disp_root
        self.load_disp = bool(load_disp)
        self.allow_inline_disp_fallback = bool(allow_inline_disp_fallback)
        if self.load_disp and (self.disp_root is None) and (not self.allow_inline_disp_fallback):
            raise ValueError(
                "[RGBNIRStereoSequenceDataset] load_disp=true requires an explicit disp_root "
                "or allow_inline_disp_fallback=true."
            )

        if frame_list_file is None or (not os.path.isfile(frame_list_file)):
            raise RuntimeError("[RGBNIRStereoSequenceDataset] frame_list_file is required (one frame directory per line).")

        frame_dirs_all: List[str] = []
        with open(frame_list_file, "r") as f:
            for ln in f:
                raw = ln.strip()
                if not raw:
                    continue
                norm = _resolve_frame_dir(self.root, raw)
                expanded = _expand_to_frame_dirs(norm)
                if not expanded:
                    continue
                frame_dirs_all.extend(expanded)

        # Bucket by sequence root and sort to preserve temporal order.
        buckets: DefaultDict[str, List[str]] = defaultdict(list)
        for fdir in frame_dirs_all:
            seq_root = os.path.dirname(_normdir(fdir))
            buckets[seq_root].append(fdir)
        for k in buckets:
            buckets[k].sort()

        # Build (t-1, t, t+1) triplets and skip boundary frames.
        self.samples: List[Dict[str, Any]] = []
        for seq_root, frames in buckets.items():
            # Read calibration once per sequence.
            try:
                K_np, b = _read_calib(seq_root)
            except Exception:
                continue

            n = len(frames)
            for i in range(self.stride, n - self.stride):
                f_tm1 = frames[i - self.stride]
                f_t   = frames[i]
                f_tp1 = frames[i + self.stride]

                # Check that all four images exist (L/R + RGB/NIR).
                def ok_frame(fdir: str) -> bool:
                    rgb_base = os.path.join(fdir, "rgb")
                    nir_base = os.path.join(fdir, "nir")
                    l_rgb = _find_one_of(rgb_base, [f"left{e}" for e in _IMAGE_EXTS])
                    r_rgb = _find_one_of(rgb_base, [f"right{e}" for e in _IMAGE_EXTS])
                    l_nir = _find_one_of(nir_base, [f"left{e}" for e in _IMAGE_EXTS])
                    r_nir = _find_one_of(nir_base, [f"right{e}" for e in _IMAGE_EXTS])
                    return bool(l_rgb and r_rgb and l_nir and r_nir)

                if not (ok_frame(f_tm1) and ok_frame(f_t) and ok_frame(f_tp1)):
                    continue

                self.samples.append(dict(
                    seq_root=seq_root,
                    frame_tm1=f_tm1,
                    frame_t=f_t,
                    frame_tp1=f_tp1,
                    frame_t_rel=(os.path.relpath(os.path.abspath(f_t), os.path.abspath(self.root))
                                 if self.disp_root else None),
                    K_left=K_np, baseline=b
                ))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[RGBNIRStereoSequenceDataset] No valid (t-1, t, t+1) triplets were found in {frame_list_file}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame_rgbn_pair(self, frame_dir: str) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        rgb_base = os.path.join(frame_dir, "rgb")
        nir_base = os.path.join(frame_dir, "nir")
        l_rgb = _find_one_of(rgb_base, [f"left{e}" for e in _IMAGE_EXTS])
        r_rgb = _find_one_of(rgb_base, [f"right{e}" for e in _IMAGE_EXTS])
        l_nir = _find_one_of(nir_base, [f"left{e}" for e in _IMAGE_EXTS])
        r_nir = _find_one_of(nir_base, [f"right{e}" for e in _IMAGE_EXTS])
        if not (l_rgb and r_rgb and l_nir and r_nir):
            raise FileNotFoundError(f"[RGBNIRStereoSequenceDataset] Missing image(s) in: {frame_dir}")
        return _load_rgb(l_rgb), _load_rgb(r_rgb), _load_nir(l_nir), _load_nir(r_nir)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.samples[idx]
        K_np = it["K_left"].copy()
        b = float(it["baseline"])
        K = torch.from_numpy(K_np).float()

        # Load t-1, t, and t+1.
        L_tm1_rgb, R_tm1_rgb, L_tm1_nir, R_tm1_nir = self._load_frame_rgbn_pair(it["frame_tm1"])
        L_t_rgb,   R_t_rgb,   L_t_nir,   R_t_nir   = self._load_frame_rgbn_pair(it["frame_t"])
        L_tp1_rgb, R_tp1_rgb, L_tp1_nir, R_tp1_nir = self._load_frame_rgbn_pair(it["frame_tp1"])

        # Optional: load precomputed pseudo disparity only when explicitly enabled.
        disp_L_np = None
        disp_R_np = None
        if self.load_disp:
            if self.disp_root and it.get("frame_t_rel", None):
                disp_L_np = _load_disp(os.path.join(self.disp_root, it["frame_t_rel"], "disp_left.npy"))
                disp_R_np = _load_disp(os.path.join(self.disp_root, it["frame_t_rel"], "disp_right.npy"))
                if self.allow_inline_disp_fallback and disp_L_np is None:
                    disp_L_np = _load_disp(os.path.join(it["frame_t"], "disp_left.npy"))
                if self.allow_inline_disp_fallback and disp_R_np is None:
                    disp_R_np = _load_disp(os.path.join(it["frame_t"], "disp_right.npy"))
            else:
                disp_L_np = _load_disp(os.path.join(it["frame_t"], "disp_left.npy"))
                disp_R_np = _load_disp(os.path.join(it["frame_t"], "disp_right.npy"))

        # RGB-only color jitter. Share parameters across all frames to preserve
        # temporal and stereo consistency.
        if self.augment and self.color_jitter:
            params = _color_jitter_params()
            L_tm1_rgb = _apply_rgb_jitter(L_tm1_rgb, params)
            R_tm1_rgb = _apply_rgb_jitter(R_tm1_rgb, params)
            L_t_rgb   = _apply_rgb_jitter(L_t_rgb, params)
            R_t_rgb   = _apply_rgb_jitter(R_t_rgb, params)
            L_tp1_rgb = _apply_rgb_jitter(L_tp1_rgb, params)
            R_tp1_rgb = _apply_rgb_jitter(R_tp1_rgb, params)

        # Resize to the target size and update K accordingly.
        H1, W1 = self.size
        W0, H0 = L_t_rgb.size
        sx, sy = W1 / W0, H1 / H0
        # Resize all frames.
        def _rz(rgb: Image.Image, nir: Image.Image) -> Tuple[Image.Image, Image.Image]:
            return rgb.resize((W1, H1), Image.BILINEAR), nir.resize((W1, H1), Image.BILINEAR)

        L_tm1_rgb, L_tm1_nir = _rz(L_tm1_rgb, L_tm1_nir)
        R_tm1_rgb, R_tm1_nir = _rz(R_tm1_rgb, R_tm1_nir)
        L_t_rgb,   L_t_nir   = _rz(L_t_rgb,   L_t_nir)
        R_t_rgb,   R_t_nir   = _rz(R_t_rgb,   R_t_nir)
        L_tp1_rgb, L_tp1_nir = _rz(L_tp1_rgb, L_tp1_nir)
        R_tp1_rgb, R_tp1_nir = _rz(R_tp1_rgb, R_tp1_nir)

        K = _resize_K(K.unsqueeze(0), sx, sy).squeeze(0)

        # Resize disparity maps.
        def _rz_disp(d_np: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            if d_np is None:
                return None
            # [H, W] -> [1, 1, H, W]
            t = torch.from_numpy(d_np).unsqueeze(0).unsqueeze(0)
            # Bilinear resize.
            t = torch.nn.functional.interpolate(t, size=(H1, W1), mode='bilinear', align_corners=False)
            # Scale disparity values with image width.
            t = t * sx
            return t.squeeze(0).squeeze(0)  # [H, W]

        disp_L_t = _rz_disp(disp_L_np)
        disp_R_t = _rz_disp(disp_R_np)

        # Horizontal flip and L/R swap for all frames in sync.
        if self.augment and random.random() < self.hflip_prob:
            # t-1
            L_tm1_rgb, R_tm1_rgb = TF.hflip(R_tm1_rgb), TF.hflip(L_tm1_rgb)
            L_tm1_nir, R_tm1_nir = TF.hflip(R_tm1_nir), TF.hflip(L_tm1_nir)
            # t
            L_t_rgb,   R_t_rgb   = TF.hflip(R_t_rgb),   TF.hflip(L_t_rgb)
            L_t_nir,   R_t_nir   = TF.hflip(R_t_nir),   TF.hflip(L_t_nir)
            # t+1
            L_tp1_rgb, R_tp1_rgb = TF.hflip(R_tp1_rgb), TF.hflip(L_tp1_rgb)
            L_tp1_nir, R_tp1_nir = TF.hflip(R_tp1_nir), TF.hflip(L_tp1_nir)

            K = _hflip_K(K.unsqueeze(0), width_after=W1).squeeze(0)

            # Flip disparity:
            # new left disparity comes from the old right map after flip,
            # and vice versa.
            if disp_R_t is not None:
                disp_L_out = TF.hflip(disp_R_t.unsqueeze(0)).squeeze(0)
            else:
                disp_L_out = None
            
            if disp_L_t is not None:
                disp_R_out = TF.hflip(disp_L_t.unsqueeze(0)).squeeze(0)
            else:
                disp_R_out = None
        else:
            # No flip.
            disp_L_out = disp_L_t
            disp_R_out = disp_R_t

        # Pack tensors.
        L_tm1_rgbn = _stack_rgbn(L_tm1_rgb, L_tm1_nir)
        L_t_rgbn   = _stack_rgbn(L_t_rgb,   L_t_nir)
        L_tp1_rgbn = _stack_rgbn(L_tp1_rgb, L_tp1_nir)
        R_t_rgbn   = _stack_rgbn(R_t_rgb,   R_t_nir)

        sample = dict(
            # Current time step used by the training loop.
            L_t_rgbn=L_t_rgbn,
            R_t_rgbn=R_t_rgbn,
            # Temporal neighbors from the left camera only.
            L_tm1_rgbn=L_tm1_rgbn,
            L_tp1_rgbn=L_tp1_rgbn,
            # Camera parameters.
            K=K,
            fx=K[0, 0].view(1),
            baseline=torch.tensor([b], dtype=torch.float32),
            # Metadata for visualization and debugging.
            meta=dict(
                seq_root=it["seq_root"],
                frame_dir_tm1=it["frame_tm1"],
                frame_dir=it["frame_t"],
                frame_dir_tp1=it["frame_tp1"],
            )
        )
        
        if disp_L_out is not None:
            sample["disp_igev_L"] = disp_L_out
            # Backward-compatible alias.
            sample["disp_igev_t"] = disp_L_out
            
        if disp_R_out is not None:
            sample["disp_igev_R"] = disp_R_out

        return sample
