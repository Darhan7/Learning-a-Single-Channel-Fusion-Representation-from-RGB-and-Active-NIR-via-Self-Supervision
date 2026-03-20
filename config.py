# -*- coding: utf-8 -*-
"""
Minimal typed config for the standalone fusion_selfsup project.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy
import os
import yaml


def _deep_set(mapping: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = mapping
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _filter_known_keys(dc_cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    known = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in payload.items() if k in known}


def _resolve_relpath(base_dir: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    if value == "":
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


@dataclass
class ModelConfig:
    dmin_px: float = 0.05
    dmax_px: float = 256.0


@dataclass
class LossConfig:
    ssim_alpha: float = 0.85
    stereo_auto_mask: bool = True
    temporal_auto_mask: bool = True


@dataclass
class DataConfig:
    root: str
    size: List[int]
    mode: str = "stereo_only"
    temporal_stride: int = 1
    batch_size: int = 4
    num_workers: int = 4
    augment: bool = True
    train_list: Optional[str] = None
    val_list: Optional[str] = None
    test_list: Optional[str] = None
    disp_root: Optional[str] = None
    load_disp: bool = False
    allow_inline_disp_fallback: bool = False
    max_samples: Optional[int] = None


@dataclass
class TrainConfig:
    mode: str = "fusion_selfsup"
    epochs: int = 50
    lr: float = 1e-4
    lr_min: float = 5e-6
    lr_scheduler: str = "cosine"
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    plateau_min_lr: float = 0.0
    weight_decay: float = 0.0
    amp: bool = True
    log_dir: str = "./exp_logs/fusion_selfsup"
    save_every: int = 100
    val_viz_every: int = 0
    save_test_viz: bool = False
    test_viz_copy_overlay_check: bool = True
    resume_ckpt: Optional[str] = None
    resume_optim: bool = False
    eval_only: bool = False
    eval_val: bool = False
    eval_test: bool = False
    eval_test_depth_sparse: bool = True
    early_stop_patience: int = 0
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 50
    strict_selfsup: bool = True
    fuse_feat_dim: int = 256
    fuse_attn_reduction: int = 4
    fuse_use_guided_filter: bool = False
    fuse_guided_radius: int = 5
    fuse_viz_unique_dir: bool = True
    lambda_fuse_align: float = 1.0
    lambda_fuse_grad: float = 0.2
    lambda_fuse_stereo_selfsup: float = 1.0
    lambda_fuse_temporal_selfsup: float = 0.0
    lambda_fuse_smooth: float = 0.002
    lambda_fuse_weight_entropy: float = 0.001

@dataclass
class FusionSelfSupConfig:
    model: ModelConfig
    loss: LossConfig
    data: DataConfig
    train: TrainConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionSelfSupConfig":
        return cls(
            model=ModelConfig(**_filter_known_keys(ModelConfig, data["model"])),
            loss=LossConfig(**_filter_known_keys(LossConfig, data.get("loss", {}))),
            data=DataConfig(**_filter_known_keys(DataConfig, data["data"])),
            train=TrainConfig(**_filter_known_keys(TrainConfig, data["train"])),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FusionSelfSupConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = cls.from_dict(data)
        cfg.resolve_paths(os.path.dirname(os.path.abspath(str(path))))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(asdict(self))

    def resolve_paths(self, base_dir: str) -> None:
        self.data.root = _resolve_relpath(base_dir, self.data.root) or self.data.root
        self.data.train_list = _resolve_relpath(base_dir, self.data.train_list)
        self.data.val_list = _resolve_relpath(base_dir, self.data.val_list)
        self.data.test_list = _resolve_relpath(base_dir, self.data.test_list)
        self.data.disp_root = _resolve_relpath(base_dir, self.data.disp_root)
        self.train.log_dir = _resolve_relpath(base_dir, self.train.log_dir) or self.train.log_dir
        self.train.resume_ckpt = _resolve_relpath(base_dir, self.train.resume_ckpt)

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        cfg = self.to_dict()
        for key, value in overrides.items():
            _deep_set(cfg, key, value)
        updated = FusionSelfSupConfig.from_dict(cfg)
        self.model = updated.model
        self.loss = updated.loss
        self.data = updated.data
        self.train = updated.train


def parse_cli_overrides(args) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.root is not None:
        overrides["data.root"] = args.root
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.device is not None:
        overrides["train.device"] = args.device
    for entry in args.set or []:
        if "=" not in entry:
            raise SystemExit(f"--set expects key=value, got: {entry}")
        k, v = entry.split("=", 1)
        overrides[k] = _smart_cast(v)
    return overrides


def _smart_cast(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    try:
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, str):
            return parsed
    except Exception:
        pass
    return raw
