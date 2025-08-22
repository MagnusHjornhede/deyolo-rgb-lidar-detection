# Thesis Experiments Log — DEYOLO on KITTI

> Master log of all experiments.  
> Status key: 🔴 planned · 🟡 running · 🟢 done · ⚠️ needs attention

---

## Environment

- OS: Windows 11, Python 3.10 (venv)
- GPU: RTX 3080 10GB (CUDA)
- Ultralytics YOLOv8 (DEYOLO head)
- Deterministic training: seed=42, AMP off unless stated
- Defaults: 100 epochs, imgsz=640, batch=16, workers=3

---

## Experiments Overview

| ID   | IR spec (Stage-2)        | Dataset YAML                                 | Run name                  | Epochs | Status | Notes |
|------|--------------------------|----------------------------------------------|---------------------------|-------:|:------:|-------|
| E1   | invd, inv, mask          | KITTI_DEYOLO_E1.yaml                         | E1_invd_inv_mask_e100     |    100 | 🔴     | Baseline 3-channel LiDAR |
| E2   | invd, log, mask          | KITTI_DEYOLO_E2.yaml                         | E2_invd_log_mask_e100     |    100 | 🔴     | Log-depth swap |
| E3   | invd ×3                  | KITTI_DEYOLO_E3.yaml                         | E3_invd3_e100             |    100 | 🔴     | Single clean depth (replicated) |
| E4   | zero IR                  | KITTI_DEYOLO_E4.yaml                         | E4_rgb_only_e100          |    100 | 🟢     | RGB-only baseline |

---
