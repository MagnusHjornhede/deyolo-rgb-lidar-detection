#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze detection predictions with an emphasis on small objects.

Inputs:
  - --pred: path to predictions.json produced by Ultralytics .val(..., save_json=True)
  - --data-yaml: dataset yaml used for training/eval (to locate test images/labels)
  - --split: dataset split to evaluate (default: test)
Outputs:
  - --out-md: Markdown summary file (default: reports/pred_analysis.md)
  - --out-csv: CSV with headline metrics (default: reports/pred_analysis.csv)
  - --plots: if set, saves simple histograms into reports/plots_*/

Assumptions:
  - Ground truth labels are in YOLO TXT format under .../labels/vis_<split>/*.txt
  - The YAML 'test' (or chosen split) points to IMAGE directory; labels dir is inferred.
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict, namedtuple
import math

import numpy as np
from PIL import Image
import yaml
import csv

# ---------------- Utils ---------------- #

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def iou_xyxy(a, b):
    # a: (N,4), b: (M,4) in xyxy
    N = a.shape[0]
    M = b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-9)

def ap_from_pr(precision, recall):
    # VOC-style 11-point or integrate? Use integrate (monotonic interpolation)
    # Sort recall ascending, precision is piecewise max envelope
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # indices where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def yolo_txt_to_gt_boxes(txt_path):
    """Read YOLO txt: each line 'cls cx cy w h' (normalized). Returns numpy arrays."""
    if not txt_path.exists():
        return np.zeros((0, 5), dtype=np.float32)  # cls + xywhn
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5 and len(parts) != 6:
                # tolerate an extra conf if present (ignore)
                if len(parts) < 5:
                    continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            rows.append([cls, cx, cy, w, h])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)

def xywhn_to_xyxy_px(xywhn, img_w, img_h):
    # xywh normalized to [0,1] -> xyxy in pixels
    cx, cy, w, h = xywhn
    x = (cx - w / 2.0) * img_w
    y = (cy - h / 2.0) * img_h
    return np.array([x, y, x + w * img_w, y + h * img_h], dtype=np.float32)

def infer_labels_dir(images_dir: Path, split: str):
    # Expect sibling "labels/vis_<split>"
    root = images_dir.parent
    labels_dir = root / "labels" / f"vis_{split}"
    return labels_dir

def area_bucket_from_gt_box_xyxy(gt_xyxy):
    # COCO buckets by *GT area* in pixels
    w = max(0.0, float(gt_xyxy[2] - gt_xyxy[0]))
    h = max(0.0, float(gt_xyxy[3] - gt_xyxy[1]))
    area = w * h
    if area < 32 * 32:
        return "small"
    elif area < 96 * 96:
        return "medium"
    else:
        return "large"

# --------------- Main eval --------------- #

def evaluate_ap50_by_class_and_size(pred_json, data_yaml, split="test", out_md=None, out_csv=None, save_plots=False):
    # Load predictions
    with open(pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)

    data = load_yaml(data_yaml)
    # names: map idx->class name
    names = data.get("names") or data.get("names_dict")
    # Ultralytics usually has names: {0: 'car', 1:'pedestrian', ...} or a list
    if isinstance(names, dict):
        # ensure integer keys order
        class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        class_names = names
    else:
        # fallback to 0..N-1 unknown names
        max_cat = max(p["category_id"] for p in preds) if preds else 2
        class_names = [f"class_{i}" for i in range(max_cat + 1)]

    # Locate images dir for split (data['test'] or ['val'] etc.)
    images_dir = Path(data.get(split) or data.get(split + "_images") or data[split])
    labels_dir = infer_labels_dir(Path(images_dir), split)

    # Group predictions by image_id and by class
    # Also need image file path map for dimensions; image_id mapping is YOLO's internal:
    # Ultralytics sets 'image_id' as the index in the dataloader order. We need to map
    # image_id->actual filename. Easiest: rely on labels .txt filenames numeric order.
    # Practical workaround: assume image_id corresponds to the numeric order Ultralytics used.
    # We'll reconstruct a mapping image_id -> image path by walking the images dir sorted.
    # This matches Ultralytics .val loader ordering (glob then sort).
    image_paths = sorted(
        [p for p in Path(images_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    id_to_img = {i: p for i, p in enumerate(image_paths)}

    # Sanity: If predictions reference higher image_id than we have, warn but continue
    max_pred_id = max((p["image_id"] for p in preds), default=-1)
    if max_pred_id >= len(image_paths):
        print(f"[WARN] predictions reference image_id up to {max_pred_id}, "
              f"but only {len(image_paths)} images found in {images_dir}.")

    # Assemble GT per image (xyxy px) and class; also bucket per GT size
    GT = defaultdict(lambda: defaultdict(list))  # GT[image_id][class_id] = [xyxy...]
    GT_buckets = defaultdict(lambda: defaultdict(list))  # size bucket parallel
    img_dims = {}  # cache (w,h)

    for img_id, img_path in id_to_img.items():
        # image size
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            # fallback size if unreadable
            w, h = 1280, 720
        img_dims[img_id] = (w, h)

        # corresponding label .txt name (same stem)
        # images could be in nested dirs; labels mirror file name in labels_dir
        txt_path = labels_dir / (img_path.stem + ".txt")
        g = yolo_txt_to_gt_boxes(txt_path)  # (N,5) -> cls, cx,cy,w,h (normalized)
        if g.size == 0:
            continue
        for row in g:
            cls = int(row[0])
            xyxy = xywhn_to_xyxy_px(row[1:], w, h)
            GT[img_id][cls].append(xyxy)
            GT_buckets[img_id][cls].append(area_bucket_from_gt_box_xyxy(xyxy))

    # Convert lists to numpy
    for img_id in GT:
        for cls in GT[img_id]:
            if GT[img_id][cls]:
                GT[img_id][cls] = np.stack(GT[img_id][cls], axis=0)
            else:
                GT[img_id][cls] = np.zeros((0, 4), dtype=np.float32)

    # Organize predictions by class
    preds_by_cls = defaultdict(list)
    for p in preds:
        cls = int(p["category_id"])
        img_id = int(p["image_id"])
        bbox = np.array(p["bbox"], dtype=np.float32)  # xywh in px
        xyxy = xywh_to_xyxy(bbox)
        score = float(p.get("score", 1.0))
        preds_by_cls[cls].append((img_id, xyxy, score))

    for cls in preds_by_cls:
        # sort by score desc
        preds_by_cls[cls].sort(key=lambda t: t[2], reverse=True)

    # Compute AP@0.5 overall + size buckets (by GT area)
    Result = namedtuple("Result", "AP AP_small AP_medium AP_large nGT nGT_s nGT_m nGT_l")
    class_results = {}
    overall_stats = {"TP": 0, "FP": 0, "FN": 0}

    def eval_one_bucket(match_mask, gt_used_mask, scores):
        # match_mask: boolean array for detections -> whether matched a GT
        # gt_used_mask: array of GT flags used (for counting FN)
        # scores: scores aligned with detections array
        if len(scores) == 0:
            return 0.0
        # Sort by score desc (already sorted)
        tp = match_mask.astype(np.float32)
        fp = (~match_mask).astype(np.float32)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / (cum_tp[-1] + (gt_used_mask.size - int(gt_used_mask.sum())))  # safe denom; corrected below
        # Better: we know total GT = gt_used_mask.size
        total_gt = gt_used_mask.size
        recall = cum_tp / (total_gt + 1e-9)
        precision = cum_tp / (cum_tp + cum_fp + 1e-9)
        return ap_from_pr(precision, recall)

    for cls_id, cls_name in enumerate(class_names):
        dets = preds_by_cls.get(cls_id, [])
        det_img_ids = np.array([d[0] for d in dets], dtype=np.int64)
        det_xyxy = np.stack([d[1] for d in dets], axis=0) if dets else np.zeros((0, 4), dtype=np.float32)
        det_scores = np.array([d[2] for d in dets], dtype=np.float32)

        # Collect all GT boxes for this class, keep per-image for matching
        gt_total = 0
        gt_small = 0
        gt_medium = 0
        gt_large = 0

        # For matching bookkeeping
        match_flags = np.zeros(det_xyxy.shape[0], dtype=bool)
        match_bucket = np.array([""], dtype=object).repeat(det_xyxy.shape[0])
        iou_thr = 0.5

        # Per-image matching (greedy)
        # To also compute bucket AP, we need to know the bucket of the GT that each det matched to.
        per_image_idx = defaultdict(list)
        for i, img_id in enumerate(det_img_ids):
            per_image_idx[img_id].append(i)

        # Count GT totals and buckets
        for img_id in id_to_img.keys():
            g = GT[img_id].get(cls_id, np.zeros((0, 4), dtype=np.float32))
            buckets = GT_buckets[img_id].get(cls_id, [])
            gt_total += g.shape[0]
            for b in buckets:
                if b == "small":   gt_small += 1
                elif b == "medium": gt_medium += 1
                elif b == "large":  gt_large += 1

            if g.shape[0] == 0:
                continue

            used = np.zeros(g.shape[0], dtype=bool)
            # Detections on this image
            idxs = per_image_idx.get(img_id, [])
            if not idxs:
                continue
            a = det_xyxy[idxs]
            ious = iou_xyxy(a, g)  # (len(idxs), n_gt)
            for local_i, det_i in enumerate(idxs):
                # best GT match
                gt_j = int(np.argmax(ious[local_i]))
                best = ious[local_i, gt_j]
                if best >= iou_thr and not used[gt_j]:
                    match_flags[det_i] = True
                    match_bucket[det_i] = buckets[gt_j]
                    used[gt_j] = True

        # Now compute AP per bucket and overall
        # overall
        AP_overall = eval_one_bucket(match_flags, np.zeros(gt_total, bool), det_scores) if gt_total > 0 else 0.0

        def ap_bucket(bucket_name, gt_count):
            if gt_count == 0:
                return 0.0
            mask = (match_bucket == bucket_name) | (match_bucket == "")  # "" for unmatched dets counts as FP in any bucket filter
            # We need to *filter detections to those that could match bucket GT OR any dets for these images?*
            # Practical approach: Only include detections whose matched GT bucket == bucket_name; plus *all* FPs are included to compute precision properly.
            # So mask above keeps TPs for this bucket and all FPs.
            return eval_one_bucket(match_flags[mask], np.zeros(gt_count, bool), det_scores[mask])

        AP_small  = ap_bucket("small",  gt_small)
        AP_medium = ap_bucket("medium", gt_medium)
        AP_large  = ap_bucket("large",  gt_large)

        class_results[cls_name] = Result(
            AP=AP_overall, AP_small=AP_small, AP_medium=AP_medium, AP_large=AP_large,
            nGT=gt_total, nGT_s=gt_small, nGT_m=gt_medium, nGT_l=gt_large
        )

    # Macro means weighted by GT count (more informative than plain mean when classes imbalanced)
    def weighted_mean(key):
        num = sum(res.__getattribute__(key) * res.nGT for res in class_results.values())
        den = sum(res.nGT for res in class_results.values()) + 1e-9
        return num / den

    summary = {
        "mAP50_weighted": weighted_mean("AP"),
        "mAP50_s_weighted": weighted_mean("AP_small"),
        "mAP50_m_weighted": weighted_mean("AP_medium"),
        "mAP50_l_weighted": weighted_mean("AP_large"),
    }

    # ---------- Save outputs ----------
    if out_md:
        ensure_dir(Path(out_md))
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# Prediction Analysis (AP@0.5)\n\n")
            f.write(f"- **Predictions:** `{pred_json}`\n")
            f.write(f"- **Data YAML:** `{data_yaml}`  |  **Split:** `{split}`\n\n")
            f.write("## Overall (GT-weighted)\n\n")
            f.write(f"- mAP@0.5 (all): **{summary['mAP50_weighted']:.3f}**\n")
            f.write(f"- mAP@0.5 (small): **{summary['mAP50_s_weighted']:.3f}**\n")
            f.write(f"- mAP@0.5 (medium): **{summary['mAP50_m_weighted']:.3f}**\n")
            f.write(f"- mAP@0.5 (large): **{summary['mAP50_l_weighted']:.3f}**\n\n")
            f.write("## Per-class\n\n")
            f.write("| class | nGT | AP@0.5 | AP_s | AP_m | AP_l |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for cls_name, r in class_results.items():
                f.write(f"| {cls_name} | {r.nGT} | {r.AP:.3f} | {r.AP_small:.3f} | {r.AP_medium:.3f} | {r.AP_large:.3f} |\n")

    if out_csv:
        ensure_dir(Path(out_csv))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "nGT", "AP50", "AP50_small", "AP50_medium", "AP50_large"])
            for cls_name, r in class_results.items():
                w.writerow([cls_name, r.nGT, f"{r.AP:.6f}", f"{r.AP_small:.6f}", f"{r.AP_medium:.6f}", f"{r.AP_large:.6f}"])
            w.writerow(["__weighted__", "", f"{summary['mAP50_weighted']:.6f}",
                        f"{summary['mAP50_s_weighted']:.6f}",
                        f"{summary['mAP50_m_weighted']:.6f}",
                        f"{summary['mAP50_l_weighted']:.6f}"])

    # Optional: very lightweight size histogram of predictions (by predicted box area)
    if save_plots:
        try:
            import matplotlib.pyplot as plt
            areas = []
            confs = []
            for p in preds:
                x, y, w, h = p["bbox"]
                areas.append(w * h)
                confs.append(p.get("score", 1.0))
            areas = np.array(areas, dtype=np.float32)
            confs = np.array(confs, dtype=np.float32)

            plot_dir = Path(out_md).with_suffix("") if out_md else Path("reports/plots_pred")
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Area histogram
            plt.figure()
            plt.hist(np.sqrt(areas), bins=50)
            plt.xlabel("sqrt(area) (px)")
            plt.ylabel("count")
            plt.title("Predicted box size distribution")
            plt.tight_layout()
            plt.savefig(plot_dir / "pred_box_size_hist.png")
            plt.close()

            # Confidence vs area scatter (subsample to 20k)
            idx = np.linspace(0, len(areas) - 1, min(len(areas), 20000)).astype(int)
            plt.figure()
            plt.scatter(np.sqrt(areas[idx]), confs[idx], s=2, alpha=0.3)
            plt.xlabel("sqrt(area) (px)")
            plt.ylabel("confidence")
            plt.title("Confidence vs predicted size")
            plt.tight_layout()
            plt.savefig(plot_dir / "pred_conf_vs_size.png")
            plt.close()
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

    return summary, class_results


def main():
    ap = argparse.ArgumentParser(description="Analyze predictions with AP@0.5 by size buckets (GT area).")
    ap.add_argument("--pred", required=True, help="Path to predictions.json")
    ap.add_argument("--data-yaml", required=True, help="Path to dataset YAML")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-md", default="reports/pred_analysis.md")
    ap.add_argument("--out-csv", default="reports/pred_analysis.csv")
    ap.add_argument("--plots", action="store_true", help="Save simple plots")
    args = ap.parse_args()

    summary, class_results = evaluate_ap50_by_class_and_size(
        args.pred, args.data_yaml, split=args.split,
        out_md=args.out_md, out_csv=args.out_csv, save_plots=args.plots
    )

    print("\n== Overall (GT-weighted) AP@0.5 ==")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    print("\nPer-class:")
    for cls, r in class_results.items():
        print(f"{cls:12s} | nGT={r.nGT:5d} | AP={r.AP:.3f} | s={r.AP_small:.3f} | m={r.AP_medium:.3f} | l={r.AP_large:.3f}")


if __name__ == "__main__":
    main()
