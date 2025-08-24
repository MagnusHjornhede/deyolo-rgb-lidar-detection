# Experiment Log

_Last updated: 2025-08-23_

## Summary Table (Test Split)

| Exp ID | Run Name                 | Notes                                   | mAP50 | mAP50-95 | Car AP50 | Ped AP50 | Cyc AP50 |
|--------|--------------------------|-----------------------------------------|-------|----------|----------|----------|----------|
| **E1** | `E1_invd_inv_mask_e100`  | IR(invd+inv+mask)+RGB                    | 0.821 | 0.542    | 0.940    | 0.728    | 0.796    |
| **E2** | `E2_invd_log_mask_e1002` | IR(logswap invd+log+mask)+RGB            | 0.841 | 0.567    | 0.941    | 0.762    | 0.821    |
| **E3** | `E3_invd3_e1002`         | IR(inv_denoised ×3)+RGB                  | 0.770 | 0.483    | 0.914    | 0.667    | 0.729    |
| **E4** | `E4_rgb_only_e1002`      | RGB-only baseline                        | 0.785 | 0.513    | 0.933    | 0.684    | 0.738    |

---
# Progress Report – KITTI + DEYOLO Experiments (E1–E4)
_Date: 2025-08-24_

## Summary
We trained and evaluated four DEYOLO variants on KITTI. The **logswap fusion (E2)** delivers the best performance overall and across all classes, with notable improvements for **Pedestrian** and **Cyclist** at **AP50–95**, indicating better localization at stricter IoUs.

## Configuration
- Model: DEYOLO (YOLOv8-based, dual-stream RGB+IR)
- Training: 100 epochs, batch 8, imgsz 640, RTX 3080 10GB
- Splits: train/val/test ≈ 4488 / 1496 / 1497
- Evaluation: Ultralytics `.val()` on test; metrics below

## Results (Test Split)

### Overall
| Exp | Input Setup | mAP@50 | mAP@50–95 |
|---|---|---:|---:|
| **E2** | RGB + logswap(inv_denoised + log + mask) | **0.841** | **0.567** |
| **E1** | RGB + inv_denoised + inv + mask | 0.821 | 0.542 |
| **E4** | RGB-only (zero IR) | 0.785 | 0.513 |
| **E3** | RGB + inv_denoised ×3 | 0.770 | 0.483 |

### Per‑class (AP50 / AP50–95)
- **Car** – E2: **0.941 / 0.746**, E1: 0.940 / 0.736, E4: 0.933 / 0.725, E3: 0.914 / 0.674  
- **Pedestrian** – E2: **0.762 / 0.431**, E1: 0.728 / 0.411, E4: 0.684 / 0.364, E3: 0.667 / 0.367  
- **Cyclist** – E2: **0.821 / 0.523**, E1: 0.796 / 0.477, E4: 0.738 / 0.451, E3: 0.729 / 0.407

## Takeaways
- **E2 (logswap fusion)** is consistently best; the margin is largest for **Pedestrian** and **Cyclist** at AP50–95 → improved small/skinny object localization.
- **E1** provides solid gains over RGB-only; **E3** (redundant inv_denoised×3) does not help.
- Resume training is finicky due to DEYOLO’s dual-stream init; we worked around it by evaluating with `best.pt` on test.

## Next Steps
1. Add **size-stratified** AP (AP_s/m/l) once image metadata is wired; current improvements suggest benefits on smaller targets.  
2. Lock these four results in repo (`EXPERIMENT_LOG.md`) and back them with predictions.json.  
3. Explore additional LiDAR encodings (e.g., normalized depth, surface normals) and ablations.

### Notes
- **E1**: Balanced baseline fusion (all three IR variants + RGB).
- **E2**: Logswap fusion seems strongest overall (best mAP50 and mAP50-95).
- **E3**: Triple inv_denoised underperformed vs E1/E2.
- **E4**: RGB-only baseline, useful as control.

All results are from **test split**, best.pt checkpoints.
## Test Results (KITTI test split)

| Exp | Input Setup | mAP@50 | mAP@50–95 |
|---|---|---:|---:|
| **E2** | RGB + logswap(inv_denoised + log + mask) | **0.841** | **0.567** |
| **E1** | RGB + inv_denoised + inv + mask | 0.821 | 0.542 |
| **E4** | RGB-only (zero IR) | 0.785 | 0.513 |
| **E3** | RGB + inv_denoised ×3 | 0.770 | 0.483 |

**Per-class AP (AP50 / AP50–95)**

- **Car**  
  - E2: **0.941 / 0.746**  
  - E1: 0.940 / 0.736  
  - E4: 0.933 / 0.725  
  - E3: 0.914 / 0.674
- **Pedestrian**  
  - E2: **0.762 / 0.431**  
  - E1: 0.728 / 0.411  
  - E4: 0.684 / 0.364  
  - E3: 0.667 / 0.367
- **Cyclist**  
  - E2: **0.821 / 0.523**  
  - E1: 0.796 / 0.477  
  - E4: 0.738 / 0.451  
  - E3: 0.729 / 0.407
---



# Experiment Log

_Auto-updated: 2025-08-24_

## Summary of Experiments (E1–E4)

All experiments trained on **KITTI-DEYOLO (RGB + LiDAR-derived 4th channel)** with 100 epochs, imgsz=640, batch=8.  
Evaluated on **test split**. Metrics are AP50-95 per class.

| Exp | IR/LiDAR Config                          | All AP50 | All AP50-95 | Car AP50-95 | Pedestrian AP50-95 | Cyclist AP50-95 | Notes |
|-----|------------------------------------------|----------|-------------|-------------|---------------------|-----------------|-------|
| **E4** | RGB-only baseline (zero 4th ch)          | 0.785    | 0.513       | 0.725       | 0.364               | 0.451           | Reference |
| **E3** | inv_denoised ×3 + RGB                   | 0.770    | 0.483       | 0.674       | 0.367               | 0.407           | Weakest small-object |
| **E2** | logswap (inv_denoised + log + mask + RGB)| 0.841    | 0.567       | 0.746       | 0.431               | 0.523           | **Best overall, strongest for Ped/Cyclist** |
| **E1** | baseline (inv_denoised + inv + mask + RGB)| 0.821    | 0.542       | 0.736       | 0.411               | 0.477           | Strong baseline |

---

## Ranked Performance (AP50-95)

### Car
1. **E2 (0.746)**  
2. E1 (0.736)  
3. E4 (0.725)  
4. E3 (0.674)  

### Pedestrian
1. **E2 (0.431)**  
2. E1 (0.411)  
3. E3 (0.367)  
4. E4 (0.364)  

### Cyclist
1. **E2 (0.523)**  
2. E1 (0.477)  
3. E4 (0.451)  
4. E3 (0.407)  

---

## Insights
- **E2 (logswap)** is consistently the **best performer** across all classes, especially for **small/tiny objects (Pedestrian, Cyclist)**.  
- **E1** is a solid baseline with competitive results, close to E2 for Cars.  
- **E3** underperforms across all metrics → likely redundant representation (inv_denoised ×3).  
- **E4 (RGB-only)** is a strong baseline, but clearly weaker on small object classes vs. LiDAR-enhanced configs.  

---
