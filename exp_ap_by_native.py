#!/usr/bin/env python3
"""AP in fixed native-height bins, so input sizes can be compared without a precision artefact.

Recall alone cannot say whether a larger input genuinely detects more small plants or merely
emits more boxes. Per-bin precision is the obvious check but is unstable at small sizes, because
a false positive has to be binned by its own predicted height and small boxes are poorly sized.
AP avoids that: it integrates the precision-recall curve over the ground truth in a bin and never
needs false positives sorted into bins.

Bins are fixed in native pixels, so the same plants populate a bin at every input size and the
comparison is paired.

    python exp_ap_by_native.py --labels-dir data/test/labels --images-dir data/test/images \
        --runs runs --tag nearfar --seeds 0 1 2 --imgsz-list 640 960 1280 1920 \
        --out results/scaling/ap_by_native.csv

AP, matching and ignore handling are imported from eval_ap, never reimplemented.
"""

import argparse
import csv
import glob
import os

import numpy as np

from eval_ap import ap_tier, box_iou, in_any, voc_ap  # noqa: F401
from eval_testset import read_gt, build_image_index, sstd

PHYS_EDGES = [16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, float("inf")]


def native_bin(h):
    for i in range(len(PHYS_EDGES) - 1):
        if PHYS_EDGES[i] <= h < PHYS_EDGES[i + 1]:
            hi = "inf" if not np.isfinite(PHYS_EDGES[i + 1]) else f"{PHYS_EDGES[i+1]:.0f}"
            return f"{PHYS_EDGES[i]:.0f}-{hi}"
    return None


def predictions_native(model, items, imgsz, a):
    """As eval_ap.predictions_for, but each GT is labelled by its native height bin."""
    per_image = []
    for ip, w, h, plants, ignores in items:
        gts = []
        for b in plants:
            nb = native_bin(b[3] - b[1])
            if nb is not None:
                gts.append((b[:4], nb))
        r = model.predict(ip, conf=a.conf, iou=0.6, imgsz=imgsz, device=a.device,
                          max_det=a.max_det, verbose=False)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            cl = r.boxes.cls.cpu().numpy()
            xy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            for j in range(len(cl)):
                if int(cl[j]) == a.plant_class:
                    preds.append((float(cf[j]), xy[j].tolist()))
        per_image.append((gts, [b[:4] for b in ignores], preds))
    return per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="nearfar")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--imgsz-list", nargs="+", type=int, default=[640, 960, 1280, 1920])
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.001, help="low, so the PR curve is complete")
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--out", default="results/scaling/ap_by_native.csv")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    idx = build_image_index(a.images_dir)
    items = []
    for lp in sorted(glob.glob(os.path.join(a.labels_dir, "*.txt"))):
        ip = idx.get(os.path.splitext(os.path.basename(lp))[0])
        if ip is None:
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        plants, ignores = read_gt(lp, w, h, a.plant_class, a.ignore_class)
        items.append((ip, w, h, plants, ignores))
    print(f"[i] {len(items)} images, AP@IoU={a.iou}, conf={a.conf}, max_det={a.max_det}")

    bins = [native_bin(PHYS_EDGES[i]) for i in range(len(PHYS_EDGES) - 1)]
    table = {}
    for imgsz in a.imgsz_list:
        per_seed = {b: [] for b in bins}
        n_gt = {}
        for s in a.seeds:
            wp = os.path.join(a.runs, f"{a.tag}_s{s}", "weights", "best.pt")
            if not os.path.exists(wp):
                print(f"  skip {a.tag} s{s}")
                continue
            pi = predictions_native(YOLO(wp), items, imgsz, a)
            for b in bins:
                v, n = ap_tier(pi, b, a.iou)
                per_seed[b].append(v)
                n_gt[b] = n
        table[imgsz] = (per_seed, n_gt)
        print(f"  imgsz {imgsz}: done")

    rows = [["imgsz", "phys_bin", "n_gt", "ap_mean", "ap_std", "n_seeds"]]
    print(f"\n  {'bin':>10s} {'n':>5s} " + "".join(f"{i:>10d}" for i in a.imgsz_list))
    for b in bins:
        cells = ""
        for imgsz in a.imgsz_list:
            per_seed, n_gt = table[imgsz]
            v = [x for x in per_seed[b] if np.isfinite(x)]
            if not v:
                cells += f"{'-':>10s}"
                continue
            m, sd = float(np.mean(v)), sstd(v)
            rows.append([imgsz, b, n_gt.get(b, 0), round(m, 4), round(sd, 4), len(v)])
            cells += f"{m:>10.3f}"
        n = table[a.imgsz_list[0]][1].get(b, 0)
        if n:
            print(f"  {b:>10s} {n:>5d} {cells}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
