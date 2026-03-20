# Our Method

This repository contains our 1-channel fusion method for pixel-aligned `RGB + NIR` inputs.

Functionality:

- Input: `RGB` and `NIR`
- Output: a fused 1-channel brightness image `F`
- Goal: preserve RGB texture details while incorporating robust NIR brightness and active-illumination cues
- Training: strict self-supervision

## Dataset

This project uses the Pixel-aligned RGB-NIR Stereo dataset introduced in:

Jinnyeong Kim and Seung-Hwan Baek, "Pixel-aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision," CVPR 2025.

Official resources:

- Project repository: <https://github.com/divisonofficer/Pixel_aligned_RGB_NIR_Stereo>
- The official repository README provides the dataset download link

The raw dataset is not redistributed in this repository. Please obtain it from the official source and place it under:

```text
datasets/pixelnir/
```

## Training

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u \
  main.py \
  --config configs/default.yaml
```

## Inference

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u \
  main.py \
  --config configs/default.yaml \
  --set train.eval_only=true \
  --set train.resume_ckpt=weights/ours/fusion_selfsup_best.pt \
  --set data.num_workers=0 \
  --set train.val_viz_every=1 \
  --set train.save_test_viz=true
```
