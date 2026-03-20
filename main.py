# -*- coding: utf-8 -*-
"""Root entrypoint for the standalone fusion_selfsup pipeline."""
from __future__ import annotations

import argparse
import os
import sys
import json

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from config import FusionSelfSupConfig, parse_cli_overrides  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default=os.path.join(_PROJ_ROOT, "configs", "default.yaml"),
    help="Path to the YAML config file",
)
parser.add_argument("--root", type=str, default=None, help="Override data.root")
parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
parser.add_argument("--device", type=str, default=None, help="Override train.device")
parser.add_argument("--set", action="append", default=[],
                    help="Override any config key with dot notation, e.g. --set data.batch_size=6")
args = parser.parse_args()

cfg = FusionSelfSupConfig.from_yaml(args.config)
overrides = parse_cli_overrides(args)
if overrides:
    cfg.apply_overrides(overrides)
cfg.resolve_paths(_PROJ_ROOT)

print("[CFG] Final fusion_selfsup config:\n" + json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
print("[Boot] importing fusion_selfsup trainer and deep-learning deps...")

from trainer import FusionSelfSupTrainer  # noqa: E402

print("[Boot] trainer import finished, starting run().")
trainer = FusionSelfSupTrainer(cfg)
trainer.run()
