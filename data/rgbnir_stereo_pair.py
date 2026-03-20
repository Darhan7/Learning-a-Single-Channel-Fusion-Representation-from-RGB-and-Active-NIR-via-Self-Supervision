# -*- coding: utf-8 -*-
"""
RGBNIRStereoPairDataset: stereo-pair dataset for (L_t, R_t) using raw RGB+NIR inputs.
- Input: a frame-directory list file, one frame_dir per line, for example:
    /root/.../09-05-17-34-36/09-05-17-34-36/17_34_37_727/
- Output: {
    L_t_rgbn: [4,H,W], R_t_rgbn: [4,H,W],
    K: [3,3], fx: [1], baseline: [1],
    meta: { seq_root, frame_dir }
  }
- Geometry augmentation: resize to target_size
- Color augmentation: RGB only; NIR only gets to_tensor(0..1) with no color jitter
- Horizontal flip: synchronize left/right flip and swap, and update K accordingly
- Calibration: prefer <seq_root>/rect/calib_rectified.npz with K_rect_left and P2,
  otherwise fall back to <seq_root>/calibration.npz
"""
from __future__ import annotations
import os
import glob
import random
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from utils.intrinsics_adapter import apply_resize_to_K as _resize_K
from utils.intrinsics_adapter import apply_hflip_to_K as _hflip_K

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

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
    return torch.cat([t_rgb, t_nir], dim=0)  # [4,H,W]

def _load_disp(path: str) -> Optional[np.ndarray]:
    if os.path.isfile(path):
        try:
            disp = np.load(path).astype(np.float32)
            disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
            disp = np.clip(disp, 0.0, 256.0)
            return disp
        except Exception:
            return None
    return None

def _read_calib(seq_root: str) -> Tuple[np.ndarray, float]:
    """
    Read K_left and baseline in meters.
    Prefer rect/calib_rectified.npz: K_rect_left and P2 -> Tx -> baseline.
    Fall back to calibration.npz: mtx_left and either P2 or R/T -> Tx.
    """
    rect_npz = os.path.join(seq_root, "rect", "calib_rectified.npz")
    calib_npz = os.path.join(seq_root, "calibration.npz")

    K = None
    baseline = None

    if os.path.isfile(rect_npz):
        d = np.load(rect_npz, allow_pickle=True)
        if "K_rect_left" in d:
            K = d["K_rect_left"].astype(np.float64)
        if "P2" in d:
            P2 = d["P2"].astype(np.float64)  # [3,4]
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
    # Convert it to meters for metric-depth evaluation.
    if float(baseline) > 10.0:
        baseline = float(baseline) * 1e-3

    return K, float(baseline)

def _color_jitter_params():
    # Use lightweight jitter and share the same parameters across L/R to keep
    # photometric consistency.
    b = 0.2 * (random.random() - 0.5) * 2 + 1.0  # [0.8, 1.2]
    c = 0.2 * (random.random() - 0.5) * 2 + 1.0
    s = 0.2 * (random.random() - 0.5) * 2 + 1.0
    h = 0.02 * (random.random() - 0.5) * 2       # [-0.02, 0.02]
    return b, c, s, h

def _apply_rgb_jitter(img: Image.Image, params) -> Image.Image:
    b, c, s, h = params
    img = TF.adjust_brightness(img, b)
    img = TF.adjust_contrast(img, c)
    img = TF.adjust_saturation(img, s)
    img = TF.adjust_hue(img, h)
    return img

class RGBNIRStereoPairDataset(Dataset):
    def __init__(
        self,
        root: str,
        size: Tuple[int, int] = (384, 512),  # (H, W)
        augment: bool = True,
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
        frame_list_file: Optional[str] = None,  # If provided, use these frame dirs exactly.
        disp_root: Optional[str] = None,        # Optional disparity-cache root.
        load_disp: bool = False,                # Strict self-supervision keeps this disabled.
        allow_inline_disp_fallback: bool = False,  # Fall back to frame_dir/disp_*.npy if cache is missing.
    ):
        super().__init__()
        self.root = root
        self.size = size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.disp_root = disp_root
        self.load_disp = bool(load_disp)
        self.allow_inline_disp_fallback = bool(allow_inline_disp_fallback)
        if self.load_disp and (self.disp_root is None) and (not self.allow_inline_disp_fallback):
            raise ValueError(
                "[RGBNIRStereoPairDataset] load_disp=true requires an explicit disp_root "
                "or allow_inline_disp_fallback=true."
            )

        # Use frame_list_file directly: one frame directory per line.
        if frame_list_file is None or (not os.path.isfile(frame_list_file)):
            raise RuntimeError("[RGBNIRStereoPairDataset] frame_list_file is required (one frame directory per line).")

        with open(frame_list_file, "r") as f:
            frame_dirs = [_resolve_frame_dir(self.root, ln.strip()) for ln in f if ln.strip()]

        self.samples: List[Dict[str, Any]] = []
        for fdir in frame_dirs:
            # Expected layout: <seq_root>/<frame>/rgb,left.png ; nir,left.png ; same for right.
            seq_root = os.path.dirname(_normdir(fdir))  # Parent directory: .../<seq>/<seq>
            try:
                K_np, b = _read_calib(seq_root)
            except Exception:
                continue

            rgb_base = os.path.join(fdir, "rgb")
            nir_base = os.path.join(fdir, "nir")
            l_rgb = _find_one_of(rgb_base, [f"left{e}" for e in _IMAGE_EXTS])
            r_rgb = _find_one_of(rgb_base, [f"right{e}" for e in _IMAGE_EXTS])
            l_nir = _find_one_of(nir_base, [f"left{e}" for e in _IMAGE_EXTS])
            r_nir = _find_one_of(nir_base, [f"right{e}" for e in _IMAGE_EXTS])
            if not (l_rgb and r_rgb and l_nir and r_nir):
                continue

            # For optional disparity cache root: compute relpath once for fast lookup.
            rel = None
            try:
                root_abs = os.path.abspath(self.root)
                f_abs = os.path.abspath(fdir)
                if os.path.commonpath([root_abs, f_abs]) == root_abs:
                    rel = os.path.relpath(f_abs, root_abs)
            except Exception:
                rel = None

            self.samples.append(dict(
                seq_root=seq_root,
                frame_dir=fdir,
                frame_rel=rel,
                left_rgb=l_rgb, right_rgb=r_rgb,
                left_nir=l_nir, right_nir=r_nir,
                K_left=K_np, baseline=b
            ))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[RGBNIRStereoPairDataset] No valid frames were found in {frame_list_file} "
                "(missing images or calibration)."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.samples[idx]
        K_np = it["K_left"].copy()
        b = float(it["baseline"])
        K = torch.from_numpy(K_np).float()  # [3,3]

        L_rgb = _load_rgb(it["left_rgb"])
        R_rgb = _load_rgb(it["right_rgb"])
        L_nir = _load_nir(it["left_nir"])
        R_nir = _load_nir(it["right_nir"])

        # Optional: load precomputed pseudo disparity only when explicitly enabled.
        frame_dir = it["frame_dir"]
        rel = it.get("frame_rel", None)
        disp_L_np = None
        disp_R_np = None
        if self.load_disp:
            if self.disp_root and rel:
                disp_L_path = os.path.join(self.disp_root, rel, "disp_left.npy")
                disp_R_path = os.path.join(self.disp_root, rel, "disp_right.npy")
                disp_L_np = _load_disp(disp_L_path)
                disp_R_np = _load_disp(disp_R_path)
                if self.allow_inline_disp_fallback and disp_L_np is None:
                    disp_L_np = _load_disp(os.path.join(frame_dir, "disp_left.npy"))
                if self.allow_inline_disp_fallback and disp_R_np is None:
                    disp_R_np = _load_disp(os.path.join(frame_dir, "disp_right.npy"))
            else:
                disp_L_np = _load_disp(os.path.join(frame_dir, "disp_left.npy"))
                disp_R_np = _load_disp(os.path.join(frame_dir, "disp_right.npy"))

        # RGB-only color jitter. Use identical parameters for L/R.
        if self.augment and self.color_jitter:
            params = _color_jitter_params()
            L_rgb = _apply_rgb_jitter(L_rgb, params)
            R_rgb = _apply_rgb_jitter(R_rgb, params)

        # Resize to the target size.
        H1, W1 = self.size
        W0, H0 = L_rgb.size
        sx, sy = W1 / W0, H1 / H0
        if abs(sx - sy) > 1e-6:
            # Another option would be aspect-preserving resize + center crop.
            # Here we simply resize the full image to a fixed target size.
            pass

        L_rgb = L_rgb.resize((W1, H1), Image.BILINEAR)
        R_rgb = R_rgb.resize((W1, H1), Image.BILINEAR)
        L_nir = L_nir.resize((W1, H1), Image.BILINEAR)
        R_nir = R_nir.resize((W1, H1), Image.BILINEAR)
        K = _resize_K(K.unsqueeze(0), sx, sy).squeeze(0)

        # resize disparity if provided (keep pixel disparity scale consistent with width resize)
        def _rz_disp(d_np: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            if d_np is None:
                return None
            t = torch.from_numpy(d_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            t = F.interpolate(t, size=(H1, W1), mode="bilinear", align_corners=False)
            t = t * sx
            return t.squeeze(0).squeeze(0)  # [H,W]

        disp_L_t = _rz_disp(disp_L_np)
        disp_R_t = _rz_disp(disp_R_np)

        # Horizontal flip and L/R swap in sync.
        if self.augment and random.random() < self.hflip_prob:
            L_rgb, R_rgb = TF.hflip(R_rgb), TF.hflip(L_rgb)
            L_nir, R_nir = TF.hflip(R_nir), TF.hflip(L_nir)
            K = _hflip_K(K.unsqueeze(0), width_after=W1).squeeze(0)

            # flip disparity:
            # New Left Disp comes from old Right (flipped)
            # New Right Disp comes from old Left (flipped)
            if disp_R_t is not None:
                disp_L_out = TF.hflip(disp_R_t.unsqueeze(0)).squeeze(0)
            else:
                disp_L_out = None
            if disp_L_t is not None:
                disp_R_out = TF.hflip(disp_L_t.unsqueeze(0)).squeeze(0)
            else:
                disp_R_out = None
        else:
            disp_L_out = disp_L_t
            disp_R_out = disp_R_t

        L_rgbn = _stack_rgbn(L_rgb, L_nir)  # [4,H,W]
        R_rgbn = _stack_rgbn(R_rgb, R_nir)

        sample = dict(
            L_t_rgbn=L_rgbn,
            R_t_rgbn=R_rgbn,
            K=K,
            fx=K[0, 0].view(1),
            baseline=torch.tensor([b], dtype=torch.float32),
            meta=dict(
                seq_root=it["seq_root"],
                frame_dir=it["frame_dir"]
            )
        )
        if disp_L_out is not None:
            sample["disp_igev_L"] = disp_L_out
            # Backward-compatible alias.
            sample["disp_igev_t"] = disp_L_out
        if disp_R_out is not None:
            sample["disp_igev_R"] = disp_R_out
        return sample
