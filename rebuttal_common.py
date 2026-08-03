#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the rebuttal experiments.

Every rebuttal script imports the paper protocol from eval_testset.py rather than reimplementing
it, so their numbers are directly comparable with the tables in the paper.
"""
import logging
import math
import os
import sys

# make sure the repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Ultralytics logs "no model scale passed" on every RT-DETR load because its yaml has no
# 'scales' dict. 'l' is correct for rtdetr-l, so the warning is harmless; drop just that line.
class _DropUltralyticsScaleWarn(logging.Filter):
    def filter(self, record):  # noqa: A003
        return "no model scale passed" not in record.getMessage()


logging.getLogger("ultralytics").addFilter(_DropUltralyticsScaleWarn())

# canonical protocol helpers; never reimplement these, or the numbers drift from the paper
from eval_testset import (  # noqa: E402
    STRATA, FINE_EDGES, IMG_EXTS, sstd, read_gt, iou_matrix, stratum, in_any,
    build_image_index,
)

REPO_ROOT = _REPO_ROOT


# ----------------------------- model loader (YOLO + RT-DETR) -----------------------------
def load_detector(weights, arch_hint=None):
    """Load a detector, routing RT-DETR vs YOLO by name; both share the predict interface."""
    name = (str(arch_hint) + " " + str(weights)).lower()
    if "rtdetr" in name or "rt-detr" in name or "detr" in name:
        from ultralytics import RTDETR
        return RTDETR(weights)
    from ultralytics import YOLO
    return YOLO(weights)


def predict_plant_boxes(model, img, imgsz, conf, device, plant_class=0, max_det=300):
    """Return (xyxy, conf) for `plant` boxes. max_det=300 matches the paper (eval_testset default)."""
    import numpy as np
    r = model.predict(img, conf=conf, iou=0.6, imgsz=imgsz, device=device,
                      max_det=max_det, verbose=False)[0]
    if r.boxes is not None and len(r.boxes):
        cl = r.boxes.cls.cpu().numpy()
        xy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        keep = cl == plant_class
        return xy[keep], cf[keep]
    return np.zeros((0, 4)), np.zeros((0,))


# ----------------------------- test items (same as eval_testset) -----------------------------
def gather_items(labels_dir, images_dir, plant_cls=0, ignore_cls=1):
    """Pair *.txt labels with images by basename -> [(img_path, w, h, plants, ignores)] and per-clip counts."""
    import glob
    import cv2
    img_idx = build_image_index(images_dir)
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    items, miss = [], []
    per_clip = {}
    for lp in label_files:
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = img_idx.get(stem)
        if ip is None:
            miss.append(stem)
            continue
        im = cv2.imread(ip)
        if im is None:
            miss.append(stem)
            continue
        h, w = im.shape[:2]
        plants, ignores = read_gt(lp, w, h, plant_cls, ignore_cls)
        items.append((ip, w, h, plants, ignores))
        clip = stem.rsplit("_f", 1)[0]
        per_clip.setdefault(clip, [0, 0])
        per_clip[clip][0] += 1
        per_clip[clip][1] += len(plants)
    return items, per_clip, miss


def greedy_hits(gt, pred, iou_thr):
    """One-to-one greedy IoU -> {gt_index: pred_index}, as in the paper protocol."""
    import numpy as np
    M = iou_matrix(gt, pred)
    gt_hit = {}
    if M.size:
        pairs = sorted(((M[gi, j], gi, j) for gi in range(len(gt)) for j in range(len(pred))
                        if M[gi, j] >= iou_thr), reverse=True)
        ug, up = set(), set()
        for _, gi, j in pairs:
            if gi in ug or j in up:
                continue
            ug.add(gi); up.add(j); gt_hit[gi] = j
    return gt_hit


# ----------------------------- statistics -----------------------------
def wilson(k, n, z=1.96):
    """Wilson confidence interval for k/n -> (p, lo, hi); used for the small far tier."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, center - half), min(1.0, center + half)


def find_checkpoints(runs_dir, tag, seeds, weight="best.pt"):
    """Existing runs/<tag>_s<seed>/weights/<weight> paths."""
    cps = [os.path.join(runs_dir, f"{tag}_s{s}", "weights", weight) for s in seeds]
    return [c for c in cps if os.path.exists(c)]
