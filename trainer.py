# -*- coding: utf-8 -*-
"""
Strict self-supervised RGB+NIR fusion trainer:
- Fusion output is always 1ch float32 in [0, 255]
- Losses are computed in normalized [0, 1]
- Geometry is auxiliary: stereo self-supervised reprojection (+ optional temporal)
"""
from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

from data_loading import build_dataloader_from_list, _to_device
from losses.photometric import ssim_l1, rgb_nir_gate, auto_masking, min_reprojection
from losses.smoothness import edge_aware_smooth
from models.weight_fusion_net import WeightFusionNet, AuxDispHead1Ch
from models.pose_resnet_2ch import PoseNet2Ch
from utils.warp import warp_stereo_rectified, warp_temporal_projective
from utils.vis import save_panel

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _gray_value(rgb: torch.Tensor) -> torch.Tensor:
    return torch.amax(rgb, dim=1, keepdim=True)


def _grad_mag(x: torch.Tensor) -> torch.Tensor:
    gx = torch.nn.functional.pad(x[..., 1:] - x[..., :-1], (1, 0, 0, 0))
    gy = torch.nn.functional.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 1, 0))
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


class FusionSelfSupTrainer:
    def __init__(self, cfg):
        self.cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
        self.device = torch.device(self.cfg["train"].get("device", "cuda"))
        self.use_amp = bool(self.cfg["train"].get("amp", False)) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.rank = 0
        self.global_step = 0

        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["train"]
        if bool(train_cfg.get("strict_selfsup", True)) and bool(data_cfg.get("load_disp", False)):
            raise ValueError("[FusionSelfSup] strict_selfsup=true does not allow data.load_disp=true.")

        self.dl_train = build_dataloader_from_list(self.cfg, "train")
        if self.dl_train is None:
            raise RuntimeError("[FusionSelfSup] train_list not found.")
        self.dl_val = build_dataloader_from_list(self.cfg, "val")
        self.dl_test = build_dataloader_from_list(self.cfg, "test")

        self.fuser = WeightFusionNet(
            feat_dim=int(train_cfg.get("fuse_feat_dim", 256)),
            reduction=int(train_cfg.get("fuse_attn_reduction", 4)),
            use_guided_filter=bool(train_cfg.get("fuse_use_guided_filter", False)),
            guided_radius=int(train_cfg.get("fuse_guided_radius", 5)),
        ).to(self.device)

        self.disp_head = AuxDispHead1Ch(
            dmin_px=float(self.cfg["model"].get("dmin_px", 0.05)),
            dmax_px=float(self.cfg["model"].get("dmax_px", 256.0)),
        ).to(self.device)

        self.use_temporal_aux = bool(train_cfg.get("lambda_fuse_temporal_selfsup", 0.0) > 0)
        self.pose = PoseNet2Ch().to(self.device) if self.use_temporal_aux else None

        params = list(self.fuser.parameters()) + list(self.disp_head.parameters())
        if self.pose is not None:
            params += list(self.pose.parameters())
        self.optim = torch.optim.AdamW(
            params,
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        self._maybe_resume_from_ckpt()

        self.alpha = float(self.cfg.get("loss", {}).get("ssim_alpha", 0.85))
        self.use_auto_mask = bool(self.cfg.get("loss", {}).get("stereo_auto_mask", True))

        self.lam_align = float(train_cfg.get("lambda_fuse_align", 1.0))
        self.lam_grad = float(train_cfg.get("lambda_fuse_grad", 0.2))
        self.lam_stereo = float(train_cfg.get("lambda_fuse_stereo_selfsup", 1.0))
        self.lam_temporal = float(train_cfg.get("lambda_fuse_temporal_selfsup", 0.0))
        self.lam_smooth = float(train_cfg.get("lambda_fuse_smooth", 0.002))
        self.lam_weight_entropy = float(train_cfg.get("lambda_fuse_weight_entropy", 0.001))

        self.log_interval = int(train_cfg.get("log_interval", 50))
        self.viz_every = int(train_cfg.get("save_every", 0))
        self.log_dir = train_cfg.get("log_dir", "./exp_logs/fusion_selfsup")
        base_viz_dir = os.path.join(self.log_dir, "viz_fusion_selfsup")
        if bool(train_cfg.get("fuse_viz_unique_dir", True)):
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.viz_dir = os.path.join(base_viz_dir, run_tag)
        else:
            self.viz_dir = base_viz_dir
        os.makedirs(self.viz_dir, exist_ok=True)

        # Optional val/test visualization. Disabled by default to avoid excessive I/O.
        self.val_viz_every = int(train_cfg.get("val_viz_every", 0))
        self.save_test_viz = bool(train_cfg.get("save_test_viz", False))
        self.test_viz_copy_overlay_check = bool(train_cfg.get("test_viz_copy_overlay_check", True))
        self.val_viz_dir = os.path.join(self.viz_dir, "val")
        self.test_viz_dir = os.path.join(self.viz_dir, "test")
        if self.val_viz_every > 0:
            os.makedirs(self.val_viz_dir, exist_ok=True)
        if self.save_test_viz:
            os.makedirs(self.test_viz_dir, exist_ok=True)
        self._did_test_viz = False

        self.eval_val = bool(train_cfg.get("eval_val", False))
        # Test-only metrics stay disabled by default to avoid repeatedly peeking at test data.
        self.eval_test = bool(train_cfg.get("eval_test", False))
        self.eval_test_depth_sparse = bool(train_cfg.get("eval_test_depth_sparse", True))

        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.latest_ckpt_path = os.path.join(self.ckpt_dir, "fusion_selfsup_latest.pt")
        self.best_ckpt_path = os.path.join(self.ckpt_dir, "fusion_selfsup_best.pt")
        self.best_val = float("inf")

        self.early_stop_patience = int(train_cfg.get("early_stop_patience", 0))
        self._no_improve = 0

        self.scheduler = None
        self._sched_name = str(train_cfg.get("lr_scheduler", "none")).lower()
        if self._sched_name == "cosine":
            base_lr = float(train_cfg.get("lr", 1e-4))
            lr_min = float(train_cfg.get("lr_min", base_lr * 0.1))
            epochs = int(train_cfg.get("epochs", 10))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=epochs, eta_min=lr_min
            )
            if self.rank == 0:
                print(f"[FusionSelfSup][scheduler] CosineAnnealingLR T_max={epochs} eta_min={lr_min}")
        elif self._sched_name == "plateau":
            factor = float(train_cfg.get("plateau_factor", 0.5))
            patience = int(train_cfg.get("plateau_patience", 3))
            min_lr = float(train_cfg.get("plateau_min_lr", 0.0))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, mode="min", factor=factor, patience=patience, min_lr=min_lr
            )
            if self.rank == 0:
                print(
                    f"[FusionSelfSup][scheduler] ReduceLROnPlateau factor={factor} patience={patience} min_lr={min_lr}"
                )

    def _maybe_resume_from_ckpt(self) -> None:
        train_cfg = self.cfg.get("train", {})
        ckpt_path = train_cfg.get("resume_ckpt", None)
        if not ckpt_path:
            return
        ckpt_path = str(ckpt_path)
        if not os.path.isfile(ckpt_path):
            if self.rank == 0:
                print(f"[FusionSelfSup][Resume] ckpt not found: {ckpt_path}")
            if bool(train_cfg.get("eval_only", False)):
                raise FileNotFoundError(
                    f"[FusionSelfSup] eval_only=true but resume_ckpt was not found: {ckpt_path}. "
                    "Use a valid checkpoint (for example fusion_selfsup_latest.pt), "
                    "otherwise export will run with random weights."
                )
            return

        try:
            obj = torch.load(ckpt_path, map_location="cpu")
            self.fuser.load_state_dict(obj.get("fuser", {}), strict=True)
            self.disp_head.load_state_dict(obj.get("disp_head", {}), strict=True)
            if self.pose is not None and obj.get("pose", None) is not None:
                self.pose.load_state_dict(obj["pose"], strict=True)
            if bool(train_cfg.get("resume_optim", False)) and obj.get("optim", None) is not None:
                try:
                    self.optim.load_state_dict(obj["optim"])
                except Exception:
                    pass
            try:
                self.global_step = int(obj.get("step", self.global_step))
            except Exception:
                pass
            if self.rank == 0:
                print(f"[FusionSelfSup][Resume] loaded: {ckpt_path} (step={self.global_step})")
        except Exception as e:
            if self.rank == 0:
                print(f"[FusionSelfSup][Resume] failed: {ckpt_path} err={type(e).__name__}: {e}")

    def _appearance_loss(self, f01: torch.Tensor, rgb01: torch.Tensor, nir01: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        v01 = _gray_value(rgb01)
        loss_v = ssim_l1(f01, v01, alpha=self.alpha)
        loss_n = ssim_l1(f01, nir01, alpha=self.alpha)
        return ((1.0 - gate) * loss_v + gate * loss_n).mean()

    def _stereo_reproj_loss(self, Il: torch.Tensor, Ir: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        ir2l, valid, _ = warp_stereo_rectified(Ir, disp)
        reproj = ssim_l1(Il, ir2l, alpha=self.alpha)
        if not self.use_auto_mask:
            return _masked_mean(reproj, valid)
        identity = ssim_l1(Il, Ir, alpha=self.alpha)
        _, am = auto_masking(identity, reproj)
        return _masked_mean(reproj, am * valid)

    def _temporal_loss(
        self,
        f_t: torch.Tensor,
        f_tm1: torch.Tensor,
        f_tp1: torch.Tensor,
        disp_t: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        if self.pose is None:
            return torch.zeros([], device=f_t.device)
        pose_tm1 = self.pose(torch.cat([f_t, f_tm1], dim=1))
        pose_tp1 = self.pose(torch.cat([f_t, f_tp1], dim=1))
        T_tm1 = self.pose.to_matrix(pose_tm1)
        T_tp1 = self.pose.to_matrix(pose_tp1)
        depth_t = 1.0 / (disp_t + 1e-6)
        w_tm1, v1, _ = warp_temporal_projective(f_tm1, depth_t, K, T_tm1)
        w_tp1, v2, _ = warp_temporal_projective(f_tp1, depth_t, K, T_tp1)

        reproj = min_reprojection([
            ssim_l1(f_t, w_tm1, alpha=self.alpha),
            ssim_l1(f_t, w_tp1, alpha=self.alpha),
        ])
        valid = (v1 + v2).clamp(0.0, 1.0)
        if not self.use_auto_mask:
            return _masked_mean(reproj, valid)

        identity = min_reprojection([
            ssim_l1(f_t, f_tm1, alpha=self.alpha),
            ssim_l1(f_t, f_tp1, alpha=self.alpha),
        ])
        _, am = auto_masking(identity, reproj)
        return _masked_mean(reproj, am * valid)

    def _weight_entropy_loss(self, weights: torch.Tensor, target: float = 0.5) -> torch.Tensor:
        entropy = -(weights.clamp(1e-6, 1.0).log() * weights).sum(dim=1, keepdim=True)
        return torch.relu(torch.tensor(float(target), device=weights.device) - entropy.mean())

    def _save_ckpt(self, path: str) -> None:
        obj = {
            "fuser": self.fuser.state_dict(),
            "disp_head": self.disp_head.state_dict(),
            "pose": (self.pose.state_dict() if self.pose is not None else None),
            "optim": self.optim.state_dict(),
            "step": int(self.global_step),
        }
        torch.save(obj, path)

    @torch.no_grad()
    def _eval_sparse_depth_metrics(self, disp_l: torch.Tensor, batch: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Test-only metric: compare predicted metric depth against sparse LiDAR depth.
        Uses per-frame `rect/depth_sparse.npy` (meters) and maps coordinates to current resized resolution.
        """
        meta = batch.get("meta", None)
        if not isinstance(meta, dict):
            return None
        frame_dirs = meta.get("frame_dir", None)
        if frame_dirs is None:
            return None

        fx = batch.get("fx", None)
        baseline = batch.get("baseline", None)
        if fx is None or baseline is None:
            return None

        fx = fx.view(-1, 1, 1, 1).float()
        baseline = baseline.view(-1, 1, 1, 1).float()
        depth_pred = (fx * baseline) / disp_l.clamp_min(1e-6)
        depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)

        sum_abs_rel = 0.0
        sum_abs = 0.0
        sum_sq = 0.0
        sum_a1 = 0.0
        n_total = 0

        bsz, _, hp, wp = depth_pred.shape
        for i in range(bsz):
            fdir = frame_dirs[i]
            if not isinstance(fdir, str):
                fdir = str(fdir)
            gt_path = os.path.join(fdir, "rect", "depth_sparse.npy")
            if not os.path.isfile(gt_path):
                continue
            gt = np.load(gt_path).astype(np.float32)
            if gt.ndim != 2:
                gt = gt.squeeze()
            if gt.ndim != 2:
                continue
            ys, xs = np.nonzero(np.isfinite(gt) & (gt > 0.0))
            if ys.size == 0:
                continue
            h0, w0 = gt.shape
            sx = float(wp) / max(float(w0), 1.0)
            sy = float(hp) / max(float(h0), 1.0)
            xs1 = np.clip(np.rint(xs * sx).astype(np.int64), 0, wp - 1)
            ys1 = np.clip(np.rint(ys * sy).astype(np.int64), 0, hp - 1)

            gt_vals = torch.from_numpy(gt[ys, xs]).to(depth_pred.device, non_blocking=True).float()
            ys1_t = torch.from_numpy(ys1).to(depth_pred.device, non_blocking=True)
            xs1_t = torch.from_numpy(xs1).to(depth_pred.device, non_blocking=True)
            pred_vals = depth_pred[i, 0, ys1_t, xs1_t].float()

            eps = 1e-6
            abs_err = (pred_vals - gt_vals).abs()
            abs_rel = abs_err / (gt_vals.abs() + eps)
            sq_err = (pred_vals - gt_vals) ** 2
            ratio = torch.maximum((pred_vals + eps) / (gt_vals + eps), (gt_vals + eps) / (pred_vals + eps))
            a1 = (ratio < 1.25).float()

            n = int(gt_vals.numel())
            n_total += n
            sum_abs += float(abs_err.sum().detach().cpu())
            sum_abs_rel += float(abs_rel.sum().detach().cpu())
            sum_sq += float(sq_err.sum().detach().cpu())
            sum_a1 += float(a1.sum().detach().cpu())

        if n_total <= 0:
            return None
        rmse = float(np.sqrt(sum_sq / max(n_total, 1)))
        return dict(
            n=int(n_total),
            abs_rel=float(sum_abs_rel / n_total),
            mae=float(sum_abs / n_total),
            rmse=rmse,
            a1=float(sum_a1 / n_total),
        )

    @torch.no_grad()
    def _eval_epoch(self, dl) -> Dict[str, float]:
        self.fuser.eval()
        self.disp_head.eval()
        if self.pose is not None:
            self.pose.eval()

        sums: Dict[str, float] = dict(
            total=0.0, align=0.0, grad=0.0, stereo=0.0, temporal=0.0, smooth=0.0, ent=0.0
        )
        n_batches = 0

        for batch in dl:
            batch = _to_device(batch, self.device)
            L = batch["L_t_rgbn"]
            R = batch["R_t_rgbn"]
            rgb_l, nir_l = L[:, :3].clamp(0.0, 1.0), L[:, 3:4].clamp(0.0, 1.0)
            rgb_r, nir_r = R[:, :3].clamp(0.0, 1.0), R[:, 3:4].clamp(0.0, 1.0)
            gate_l = rgb_nir_gate(torch.cat([rgb_l, nir_l], dim=1)).detach()
            gate_r = rgb_nir_gate(torch.cat([rgb_r, nir_r], dim=1)).detach()

            with autocast(enabled=self.use_amp):
                out_l = self.fuser(rgb_l, nir_l)
                out_r = self.fuser(rgb_r, nir_r)
                f_l01 = out_l["f01"]
                f_r01 = out_r["f01"]
                disp_l = self.disp_head(f_l01)
                disp_r = self.disp_head(f_r01)

                loss_align = 0.5 * (
                    self._appearance_loss(f_l01, rgb_l, nir_l, gate_l)
                    + self._appearance_loss(f_r01, rgb_r, nir_r, gate_r)
                )

                grad_ref_l = torch.maximum(_grad_mag(_gray_value(rgb_l)), _grad_mag(nir_l))
                grad_ref_r = torch.maximum(_grad_mag(_gray_value(rgb_r)), _grad_mag(nir_r))
                loss_grad = 0.5 * (
                    (_grad_mag(f_l01) - grad_ref_l).abs().mean()
                    + (_grad_mag(f_r01) - grad_ref_r).abs().mean()
                )

                loss_stereo = 0.5 * (
                    self._stereo_reproj_loss(f_l01, f_r01, disp_l)
                    + self._stereo_reproj_loss(f_r01, f_l01, disp_r)
                )
                loss_smooth = 0.5 * (edge_aware_smooth(disp_l, f_l01) + edge_aware_smooth(disp_r, f_r01))
                loss_entropy = 0.5 * (
                    self._weight_entropy_loss(out_l["weights"]) + self._weight_entropy_loss(out_r["weights"])
                )

                loss_temporal = torch.zeros([], device=self.device)
                if self.lam_temporal > 0 and self.pose is not None and "L_tm1_rgbn" in batch and "L_tp1_rgbn" in batch:
                    Ltm1 = batch["L_tm1_rgbn"]
                    Ltp1 = batch["L_tp1_rgbn"]
                    rgb_tm1, nir_tm1 = Ltm1[:, :3].clamp(0.0, 1.0), Ltm1[:, 3:4].clamp(0.0, 1.0)
                    rgb_tp1, nir_tp1 = Ltp1[:, :3].clamp(0.0, 1.0), Ltp1[:, 3:4].clamp(0.0, 1.0)
                    out_tm1 = self.fuser(rgb_tm1, nir_tm1)
                    out_tp1 = self.fuser(rgb_tp1, nir_tp1)
                    loss_temporal = self._temporal_loss(
                        f_t=f_l01,
                        f_tm1=out_tm1["f01"],
                        f_tp1=out_tp1["f01"],
                        disp_t=disp_l,
                        K=batch["K"],
                    )

                loss = (
                    self.lam_align * loss_align
                    + self.lam_grad * loss_grad
                    + self.lam_stereo * loss_stereo
                    + self.lam_temporal * loss_temporal
                    + self.lam_smooth * loss_smooth
                    + self.lam_weight_entropy * loss_entropy
                )

            if not torch.isfinite(loss):
                continue
            n_batches += 1
            sums["total"] += float(loss.detach().float().cpu())
            sums["align"] += float(loss_align.detach().float().cpu())
            sums["grad"] += float(loss_grad.detach().float().cpu())
            sums["stereo"] += float(loss_stereo.detach().float().cpu())
            sums["temporal"] += float(loss_temporal.detach().float().cpu())
            sums["smooth"] += float(loss_smooth.detach().float().cpu())
            sums["ent"] += float(loss_entropy.detach().float().cpu())

        for k in list(sums.keys()):
            sums[k] = sums[k] / max(n_batches, 1)

        self.fuser.train()
        self.disp_head.train()
        if self.pose is not None:
            self.pose.train()
        return sums

    def _frame_rel_for_viz(self, frame_dir: str) -> str:
        """
        Map absolute frame_dir to a stable relative path under data.root for viz output.
        Falls back to a compact name if relpath cannot be computed.
        """
        root = str(self.cfg.get("data", {}).get("root", "") or "")
        try:
            root_abs = os.path.abspath(root)
            frame_abs = os.path.abspath(str(frame_dir))
            if root_abs and os.path.commonpath([root_abs, frame_abs]) == root_abs:
                rel = os.path.relpath(frame_abs, root_abs)
                rel = rel.lstrip(os.sep)
                if rel and (not rel.startswith("..")):
                    return rel
        except Exception:
            pass
        try:
            parts = os.path.normpath(str(frame_dir)).split(os.sep)
            tail = parts[-3:] if len(parts) >= 3 else parts[-1:]
            return "__".join([p for p in tail if p])
        except Exception:
            return "unknown_frame"

    @torch.no_grad()
    def _save_viz_bundle(
        self,
        out_dir: str,
        *,
        rgb01: torch.Tensor,
        nir01: torch.Tensor,
        f255: torch.Tensor,
        w_v: torch.Tensor,
        w_n: torch.Tensor,
        panel_name: str,
        fusion_name: str,
        overlay_check_src: Optional[str] = None,
    ) -> None:
        os.makedirs(out_dir, exist_ok=True)

        rgb = rgb01.detach().cpu().clamp(0.0, 1.0)
        nir = nir01.detach().cpu().clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        fuse01 = (f255.detach().cpu() / 255.0).clamp(0.0, 1.0)
        fuse = fuse01.repeat(1, 3, 1, 1)
        w_v = w_v.detach().cpu().clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        w_n = w_n.detach().cpu().clamp(0.0, 1.0).repeat(1, 3, 1, 1)

        panel = torch.cat([rgb, nir, fuse, w_v, w_n], dim=0)
        save_panel([os.path.join(out_dir, panel_name)], [panel], nrow=5)
        # save as 1ch grayscale (closer to the actual fusion output semantics)
        save_panel([os.path.join(out_dir, fusion_name)], [fuse01], nrow=1)

        if overlay_check_src and os.path.isfile(overlay_check_src):
            dst = os.path.join(out_dir, "overlay_check.png")
            if not os.path.isfile(dst):
                try:
                    shutil.copyfile(overlay_check_src, dst)
                except Exception:
                    pass

    @torch.no_grad()
    def _viz_val_epoch(self, ep: int) -> None:
        if self.val_viz_every <= 0 or self.dl_val is None:
            return

        self.fuser.eval()
        self.disp_head.eval()

        for it, batch in enumerate(self.dl_val, start=1):
            if it % self.val_viz_every != 0:
                continue
            batch = _to_device(batch, self.device)
            L = batch["L_t_rgbn"]
            rgb_l, nir_l = L[:, :3].clamp(0.0, 1.0), L[:, 3:4].clamp(0.0, 1.0)

            with autocast(enabled=self.use_amp):
                out_l = self.fuser(rgb_l, nir_l)
                disp_l = self.disp_head(out_l["f01"])

            # only save the first sample for val (keeps I/O bounded even if bs>1)
            meta = batch.get("meta", {}) if isinstance(batch, dict) else {}
            frame_dir = None
            if isinstance(meta, dict) and "frame_dir" in meta:
                try:
                    frame_dir = meta["frame_dir"][0]
                except Exception:
                    frame_dir = meta.get("frame_dir", None)
            rel = self._frame_rel_for_viz(str(frame_dir) if frame_dir is not None else "unknown_frame")
            out_dir = os.path.join(self.val_viz_dir, rel)

            tag = f"ep{int(ep):03d}_it{int(it):05d}_step{int(self.global_step):07d}"
            self._save_viz_bundle(
                out_dir,
                rgb01=rgb_l[:1],
                nir01=nir_l[:1],
                f255=out_l["f255"][:1],
                w_v=out_l["w_v"][:1],
                w_n=out_l["w_n"][:1],
                panel_name=f"panel_{tag}.png",
                fusion_name=f"fusion_{tag}.png",
            )

        self.fuser.train()
        self.disp_head.train()

    @torch.no_grad()
    def _viz_test_all(self) -> None:
        if (not self.save_test_viz) or self.dl_test is None:
            return

        self.fuser.eval()
        self.disp_head.eval()

        for batch in self.dl_test:
            batch = _to_device(batch, self.device)
            L = batch["L_t_rgbn"]
            rgb_l, nir_l = L[:, :3].clamp(0.0, 1.0), L[:, 3:4].clamp(0.0, 1.0)

            with autocast(enabled=self.use_amp):
                out_l = self.fuser(rgb_l, nir_l)
                disp_l = self.disp_head(out_l["f01"])

            meta = batch.get("meta", {})
            frame_dirs = None
            if isinstance(meta, dict):
                frame_dirs = meta.get("frame_dir", None)

            bsz = int(L.shape[0])
            for i in range(bsz):
                fdir = None
                if isinstance(frame_dirs, (list, tuple)) and i < len(frame_dirs):
                    fdir = frame_dirs[i]
                elif isinstance(frame_dirs, str):
                    fdir = frame_dirs
                rel = self._frame_rel_for_viz(str(fdir) if fdir is not None else f"unknown_frame_{i:03d}")
                out_dir = os.path.join(self.test_viz_dir, rel)

                overlay_src = None
                if self.test_viz_copy_overlay_check and fdir is not None:
                    overlay_src = os.path.join(str(fdir), "rect", "overlay_check.png")

                self._save_viz_bundle(
                    out_dir,
                    rgb01=rgb_l[i : i + 1],
                    nir01=nir_l[i : i + 1],
                    f255=out_l["f255"][i : i + 1],
                    w_v=out_l["w_v"][i : i + 1],
                    w_n=out_l["w_n"][i : i + 1],
                    panel_name="panel.png",
                    fusion_name="fusion.png",
                    overlay_check_src=overlay_src,
                )

        self.fuser.train()
        self.disp_head.train()

    @torch.no_grad()
    def _viz(
        self,
        step: int,
        rgb: torch.Tensor,
        nir: torch.Tensor,
        f255: torch.Tensor,
        w_v: torch.Tensor,
        w_n: torch.Tensor,
    ) -> None:
        if self.viz_every <= 0 or step % self.viz_every != 0:
            return
        rgb = rgb[:1].detach().cpu().clamp(0.0, 1.0)
        nir = nir[:1].detach().cpu().clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        fuse = (f255[:1].detach().cpu() / 255.0).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        w_v = w_v[:1].detach().cpu().clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        w_n = w_n[:1].detach().cpu().clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        panel = torch.cat([rgb, nir, fuse, w_v, w_n], dim=0)
        out_path = os.path.join(self.viz_dir, f"viz_{step:07d}.png")
        save_panel([out_path], [panel], nrow=5)

    def run(self):
        epochs = int(self.cfg["train"].get("epochs", 1))
        eval_only = bool(self.cfg["train"].get("eval_only", False))

        if eval_only:
            if self.rank == 0:
                print("[FusionSelfSup] eval_only=true: skip training, only run val/test viz + optional metrics.")
            if self.val_viz_every > 0 and self.dl_val is not None and self.rank == 0:
                self._viz_val_epoch(ep=0)
            if self.eval_val and self.dl_val is not None:
                val_sums = self._eval_epoch(self.dl_val)
                if self.rank == 0:
                    msg = " ".join([f"{k}={v:.6f}" for k, v in val_sums.items()])
                    print(f"[FusionSelfSup][Val][eval_only] {msg}")
            if self.save_test_viz and self.dl_test is not None and self.rank == 0:
                self._viz_test_all()
                self._did_test_viz = True
            if self.eval_test and self.dl_test is not None and self.eval_test_depth_sparse:
                self.fuser.eval()
                self.disp_head.eval()
                agg = dict(n=0, abs_rel=0.0, mae=0.0, rmse_sumsq=0.0, a1=0.0)
                for batch in self.dl_test:
                    batch = _to_device(batch, self.device)
                    L = batch["L_t_rgbn"]
                    rgb_l, nir_l = L[:, :3].clamp(0.0, 1.0), L[:, 3:4].clamp(0.0, 1.0)
                    with autocast(enabled=self.use_amp):
                        out_l = self.fuser(rgb_l, nir_l)
                        disp_l = self.disp_head(out_l["f01"])
                    mets = self._eval_sparse_depth_metrics(disp_l, batch)
                    if mets is None:
                        continue
                    n = int(mets["n"])
                    agg["n"] += n
                    agg["abs_rel"] += float(mets["abs_rel"]) * n
                    agg["mae"] += float(mets["mae"]) * n
                    agg["rmse_sumsq"] += float(mets["rmse"]) ** 2 * n
                    agg["a1"] += float(mets["a1"]) * n
                self.fuser.train()
                self.disp_head.train()
                if self.rank == 0 and agg["n"] > 0:
                    n = float(agg["n"])
                    abs_rel = agg["abs_rel"] / n
                    mae = agg["mae"] / n
                    rmse = float(np.sqrt(agg["rmse_sumsq"] / n))
                    a1 = agg["a1"] / n
                    print(
                        "[FusionSelfSup][Test][eval_only] sparse_depth "
                        f"n={int(n)} abs_rel={abs_rel:.6f} mae={mae:.6f} rmse={rmse:.6f} a1={a1:.6f}"
                    )
            print("[FusionSelfSup] done.")
            return

        for ep in range(1, epochs + 1):
            dl_len = len(self.dl_train)
            if tqdm is not None:
                pbar = tqdm(total=dl_len, desc=f"FusionSelfSup ep{ep}")
            else:
                pbar = None

            data_iter = iter(self.dl_train)
            for batch_idx in range(dl_len):
                t_fetch0 = time.perf_counter()
                batch = next(data_iter)
                t_fetch1 = time.perf_counter()
                if self.global_step == 0 and self.rank == 0:
                    print(f"[FusionSelfSup][startup] first batch fetched in {t_fetch1 - t_fetch0:.3f}s")

                self.global_step += 1
                t_step0 = time.perf_counter()
                batch = _to_device(batch, self.device)
                L = batch["L_t_rgbn"]
                R = batch["R_t_rgbn"]
                rgb_l, nir_l = L[:, :3].clamp(0.0, 1.0), L[:, 3:4].clamp(0.0, 1.0)
                rgb_r, nir_r = R[:, :3].clamp(0.0, 1.0), R[:, 3:4].clamp(0.0, 1.0)
                gate_l = rgb_nir_gate(torch.cat([rgb_l, nir_l], dim=1)).detach()
                gate_r = rgb_nir_gate(torch.cat([rgb_r, nir_r], dim=1)).detach()

                self.optim.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    out_l = self.fuser(rgb_l, nir_l)
                    out_r = self.fuser(rgb_r, nir_r)

                    f_l255 = out_l["f255"]
                    f_r255 = out_r["f255"]
                    f_l01 = out_l["f01"]
                    f_r01 = out_r["f01"]

                    disp_l = self.disp_head(f_l01)
                    disp_r = self.disp_head(f_r01)

                    loss_align_l = self._appearance_loss(f_l01, rgb_l, nir_l, gate_l)
                    loss_align_r = self._appearance_loss(f_r01, rgb_r, nir_r, gate_r)
                    loss_align = 0.5 * (loss_align_l + loss_align_r)

                    grad_ref_l = torch.maximum(_grad_mag(_gray_value(rgb_l)), _grad_mag(nir_l))
                    grad_ref_r = torch.maximum(_grad_mag(_gray_value(rgb_r)), _grad_mag(nir_r))
                    loss_grad = 0.5 * (
                        (_grad_mag(f_l01) - grad_ref_l).abs().mean()
                        + (_grad_mag(f_r01) - grad_ref_r).abs().mean()
                    )

                    loss_stereo = 0.5 * (
                        self._stereo_reproj_loss(f_l01, f_r01, disp_l)
                        + self._stereo_reproj_loss(f_r01, f_l01, disp_r)
                    )
                    loss_smooth = 0.5 * (
                        edge_aware_smooth(disp_l, f_l01) + edge_aware_smooth(disp_r, f_r01)
                    )
                    loss_entropy = 0.5 * (
                        self._weight_entropy_loss(out_l["weights"]) + self._weight_entropy_loss(out_r["weights"])
                    )

                    loss_temporal = torch.zeros([], device=self.device)
                    if self.lam_temporal > 0 and self.pose is not None and "L_tm1_rgbn" in batch and "L_tp1_rgbn" in batch:
                        Ltm1 = batch["L_tm1_rgbn"]
                        Ltp1 = batch["L_tp1_rgbn"]
                        rgb_tm1, nir_tm1 = Ltm1[:, :3].clamp(0.0, 1.0), Ltm1[:, 3:4].clamp(0.0, 1.0)
                        rgb_tp1, nir_tp1 = Ltp1[:, :3].clamp(0.0, 1.0), Ltp1[:, 3:4].clamp(0.0, 1.0)
                        out_tm1 = self.fuser(rgb_tm1, nir_tm1)
                        out_tp1 = self.fuser(rgb_tp1, nir_tp1)
                        loss_temporal = self._temporal_loss(
                            f_t=f_l01,
                            f_tm1=out_tm1["f01"],
                            f_tp1=out_tp1["f01"],
                            disp_t=disp_l,
                            K=batch["K"],
                        )

                    loss = (
                        self.lam_align * loss_align
                        + self.lam_grad * loss_grad
                        + self.lam_stereo * loss_stereo
                        + self.lam_temporal * loss_temporal
                        + self.lam_smooth * loss_smooth
                        + self.lam_weight_entropy * loss_entropy
                    )

                if not torch.isfinite(loss):
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                self._viz(self.global_step, rgb_l, nir_l, f_l255, out_l["w_v"], out_l["w_n"])

                if pbar is not None:
                    pbar.set_postfix(
                        loss=float(loss.detach().cpu()),
                        align=float(loss_align.detach().cpu()),
                        grad=float(loss_grad.detach().cpu()),
                        st=float(loss_stereo.detach().cpu()),
                        sm=float(loss_smooth.detach().cpu()),
                        tp=float(loss_temporal.detach().cpu()),
                    )
                    pbar.update(1)

                if self.global_step == 1 and self.rank == 0:
                    print(f"[FusionSelfSup][startup] first step finished in {time.perf_counter() - t_step0:.3f}s")

                if self.rank == 0 and self.global_step % self.log_interval == 0:
                    print(
                        f"[FusionSelfSup] step{self.global_step} "
                        f"loss={loss.detach().item():.4f} align={loss_align.detach().item():.4f} "
                        f"grad={loss_grad.detach().item():.4f} stereo={loss_stereo.detach().item():.4f} "
                        f"temp={loss_temporal.detach().item():.4f} smooth={loss_smooth.detach().item():.4f} "
                        f"ent={loss_entropy.detach().item():.4f} "
                        f"f255=[{f_l255.detach().amin().item():.2f},{f_l255.detach().amax().item():.2f}]"
                    )

            if pbar is not None:
                pbar.close()

            # --- epoch-end eval / ckpt / scheduler ---
            if self.val_viz_every > 0 and self.dl_val is not None and self.rank == 0:
                self._viz_val_epoch(ep)

            val_metric = None
            if self.eval_val and self.dl_val is not None:
                val_sums = self._eval_epoch(self.dl_val)
                val_metric = float(val_sums["total"])
                if self.rank == 0:
                    msg = " ".join([f"{k}={v:.6f}" for k, v in val_sums.items()])
                    print(f"[FusionSelfSup][Val][ep{ep}] {msg}")

                if val_metric < self.best_val:
                    self.best_val = val_metric
                    self._no_improve = 0
                    if self.rank == 0:
                        self._save_ckpt(self.best_ckpt_path)
                        print(f"[FusionSelfSup][Saver] best -> {self.best_ckpt_path} (val_total={val_metric:.6f})")
                else:
                    self._no_improve += 1

            if self.rank == 0:
                self._save_ckpt(self.latest_ckpt_path)

            if self.scheduler is not None:
                if self._sched_name == "plateau":
                    if val_metric is not None:
                        self.scheduler.step(val_metric)
                else:
                    self.scheduler.step()
                if self.rank == 0:
                    lr = self.optim.param_groups[0]["lr"]
                    print(f"[FusionSelfSup][scheduler] ep{ep} lr -> {lr:.6e}")

            if self.eval_test and self.dl_test is not None and self.eval_test_depth_sparse:
                self.fuser.eval()
                self.disp_head.eval()
                agg = dict(n=0, abs_rel=0.0, mae=0.0, rmse_sumsq=0.0, a1=0.0)
                for batch in self.dl_test:
                    batch = _to_device(batch, self.device)
                    L = batch["L_t_rgbn"]
                    rgb_l, nir_l = L[:, :3].clamp(0.0, 1.0), L[:, 3:4].clamp(0.0, 1.0)
                    with autocast(enabled=self.use_amp):
                        out_l = self.fuser(rgb_l, nir_l)
                        disp_l = self.disp_head(out_l["f01"])
                    mets = self._eval_sparse_depth_metrics(disp_l, batch)
                    if mets is None:
                        continue
                    n = int(mets["n"])
                    agg["n"] += n
                    agg["abs_rel"] += float(mets["abs_rel"]) * n
                    agg["mae"] += float(mets["mae"]) * n
                    agg["rmse_sumsq"] += float(mets["rmse"]) ** 2 * n
                    agg["a1"] += float(mets["a1"]) * n

                self.fuser.train()
                self.disp_head.train()

                if self.rank == 0 and agg["n"] > 0:
                    n = float(agg["n"])
                    abs_rel = agg["abs_rel"] / n
                    mae = agg["mae"] / n
                    rmse = float(np.sqrt(agg["rmse_sumsq"] / n))
                    a1 = agg["a1"] / n
                    print(
                        f"[FusionSelfSup][Test][ep{ep}] sparse_depth "
                        f"n={int(n)} abs_rel={abs_rel:.6f} mae={mae:.6f} rmse={rmse:.6f} a1={a1:.6f}"
                    )

            # test viz is extremely I/O heavy; run once at the very end
            if self.save_test_viz and self.dl_test is not None and self.rank == 0 and ep == epochs:
                self._viz_test_all()
                self._did_test_viz = True

            if self.early_stop_patience > 0 and self._no_improve >= self.early_stop_patience:
                if self.rank == 0:
                    print(f"[FusionSelfSup][EarlyStop] patience={self.early_stop_patience} stop at ep{ep}")
                break

        if self.save_test_viz and (not self._did_test_viz) and self.dl_test is not None and self.rank == 0:
            self._viz_test_all()
            self._did_test_viz = True

        print("[FusionSelfSup] done.")
