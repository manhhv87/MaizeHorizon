#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is h50 a constant, or does it depend on plant-soil contrast?

Johnson-style criteria depend on target-background contrast, and maize-soil contrast varies with
soil moisture and illumination. This computes ExG (2G-R-B) contrast between the inside and the
surround of every ground-truth box, splits plants into terciles, and refits h50 per tercile using
the same h50() and PHYS_EDGES as exp_resolution_sweep.py.

  <out>_by_contrast.csv  recall by physical-height bin, split by contrast tercile
  <out>_h50.csv          h50 per tercile, with near/mid/far recall
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuttal_common import gather_items, greedy_hits, load_detector, predict_plant_boxes  # noqa: E402
from exp_resolution_sweep import h50, PHYS_EDGES, bin_index  # noqa: E402

STRATA = ["near", "mid", "far", "all"]


def exg_contrast(im, box, other_boxes, margin=1.0):
    """ExG (2G-R-B) contrast: mean inside the box minus mean of the surround, excluding other plants.
    Higher means the plant stands out more from the soil."""
    H, W = im.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1 = max(x1, 0); y1 = max(y1, 0); x2 = min(x2, W); y2 = min(y2, H)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return float("nan")
    b = im.astype(np.float32)
    exg = 2.0 * b[:, :, 1] - b[:, :, 0] - b[:, :, 2]     # BGR: G - R - B... (2G-R-B)
    inside = exg[y1:y2, x1:x2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(round(bw * margin)), int(round(bh * margin))
    rx1, ry1 = max(x1 - mx, 0), max(y1 - my, 0)
    rx2, ry2 = min(x2 + mx, W), min(y2 + my, H)
    ring_mask = np.ones((ry2 - ry1, rx2 - rx1), bool)
    ring_mask[(y1 - ry1):(y2 - ry1), (x1 - rx1):(x2 - rx1)] = False   # bo phan trong box
    # exclude other plants from the surround
    for ob in other_boxes:
        ox1, oy1, ox2, oy2 = [int(round(v)) for v in ob[:4]]
        ox1 = max(ox1, rx1); oy1 = max(oy1, ry1); ox2 = min(ox2, rx2); oy2 = min(oy2, ry2)
        if ox2 > ox1 and oy2 > oy1:
            ring_mask[(oy1 - ry1):(oy2 - ry1), (ox1 - rx1):(ox2 - rx1)] = False
    ring = exg[ry1:ry2, rx1:rx2][ring_mask]
    if inside.size < 4 or ring.size < 4:
        return float("nan")
    return float(inside.mean() - ring.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--weights", required=True, help="one or more .pt files, comma separated (pooled over seeds)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--ring-margin", type=float, default=1.0)
    ap.add_argument("--out", default="rebuttal_contrast")
    a = ap.parse_args()

    import cv2
    items, per_clip, miss = gather_items(a.labels_dir, a.images_dir, a.plant_class, a.ignore_class)
    if not items:
        raise SystemExit("No label could be paired with an image.")
    cps = [c for c in a.weights.split(",") if os.path.exists(c)]
    if not cps:
        raise SystemExit("No checkpoint found in --weights.")
    print(f"[i] {len(items)} images | {len(cps)} seeds | imgsz={a.imgsz} iou={a.iou} conf={a.conf}")

    # 1) contrast and physical height per GT plant (model-independent)
    per_img = []   # (ip,w,h,gt,ig, contrasts[list], hphys[list])
    all_contrast = []
    for ip, w, h, gt, ig in items:
        im = cv2.imread(ip)
        cons, hph = [], []
        for gi, gb in enumerate(gt):
            others = [g for k, g in enumerate(gt) if k != gi]
            c = exg_contrast(im, gb, others, a.ring_margin)
            cons.append(c); hph.append(gb[3] - gb[1])
            if not np.isnan(c):
                all_contrast.append(c)
        per_img.append((ip, w, h, gt, ig, cons, hph))
    if len(all_contrast) < 6:
        raise SystemExit("Too few GT boxes with a valid contrast value.")
    q1, q2 = np.percentile(all_contrast, [33.3, 66.7])
    print(f"[i] tuong phan ExG: terciles tai {q1:.1f} / {q2:.1f} "
          f"(low<{q1:.1f}, {q1:.1f}<=mid<{q2:.1f}, high>={q2:.1f})")

    def tercile(c):
        if np.isnan(c):
            return None
        return "low" if c < q1 else ("mid" if c < q2 else "high")

    # 2) run the detectors and count hits per (tercile x physical-height bin)
    TERCS = ["low", "mid", "high"]
    nb = len(PHYS_EDGES) - 1
    det = {t: np.zeros(nb) for t in TERCS}
    tot = {t: np.zeros(nb) for t in TERCS}
    tier_det = {t: {s: 0 for s in STRATA} for t in TERCS}
    tier_tot = {t: {s: 0 for s in STRATA} for t in TERCS}
    for cp in cps:
        model = load_detector(cp, cp)
        for ip, w, h, gt, ig, cons, hph in per_img:
            scale = a.imgsz / max(w, h)
            pred, _ = predict_plant_boxes(model, ip, a.imgsz, a.conf, a.device, a.plant_class)
            hit = greedy_hits(gt, pred, a.iou)
            for gi, gb in enumerate(gt):
                t = tercile(cons[gi])
                if t is None:
                    continue
                hh = hph[gi]; is_hit = int(gi in hit)
                bj = bin_index(PHYS_EDGES, hh)
                det[t][bj] += is_hit; tot[t][bj] += 1
                pot = hh * scale
                s = "near" if pot >= 32 else ("mid" if pot >= 16 else "far")
                tier_det[t][s] += is_hit; tier_tot[t][s] += 1
                tier_det[t]["all"] += is_hit; tier_tot[t]["all"] += 1

    # 3) write results and fit h50 per tercile
    rows = [["contrast_tercile", "phys_lo", "phys_hi", "n_gt", "recall"]]
    h50_rows = [["contrast_tercile", "h50_phys_px", "recall_near", "recall_mid", "recall_far", "n_gt"]]
    print("\n=== h50 by contrast tercile (physical height, native px) ===")
    for t in TERCS:
        with np.errstate(invalid="ignore", divide="ignore"):
            rec = np.where(tot[t] > 0, det[t] / np.maximum(tot[t], 1), np.nan)
        for b in range(nb):
            lo, hi = PHYS_EDGES[b], PHYS_EDGES[b + 1]
            hi_s = "inf" if not np.isfinite(hi) else f"{hi:.0f}"
            rows.append([t, f"{lo:.0f}", hi_s, int(tot[t][b]),
                         "" if np.isnan(rec[b]) else round(float(rec[b]), 4)])
        hv = h50(PHYS_EDGES, rec)
        tr = {s: (tier_det[t][s] / tier_tot[t][s] if tier_tot[t][s] else float("nan")) for s in STRATA}
        h50_rows.append([t, round(hv, 2) if np.isfinite(hv) else "nan",
                         round(tr["near"], 4), round(tr["mid"], 4), round(tr["far"], 4),
                         int(tier_tot[t]["all"])])
        print(f"  contrast {t:4}: h50 ~ {hv:5.1f}px  | near/mid/far = "
              f"{tr['near']:.3f}/{tr['mid']:.3f}/{tr['far']:.3f}  (n={int(tier_tot[t]['all'])})")

    with open(a.out + "_by_contrast.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    with open(a.out + "_h50.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(h50_rows)
    print(f"\n-> {a.out}_by_contrast.csv\n-> {a.out}_h50.csv")
    h_low = next((r[1] for r in h50_rows[1:] if r[0] == "low"), "nan")
    h_high = next((r[1] for r in h50_rows[1:] if r[0] == "high"), "nan")
    print(f"\n# if h50(low) clearly exceeds h50(high), h50 is a contrast-dependent range, "
          f"KHONG phai hang so. Cap nhat cau 'h50~52px' trong paper thanh 'h50 ~ [x,y]px tuy tuong phan'.")
    print(f"#   h50(low-contrast)={h_low}px  vs  h50(high-contrast)={h_high}px")


if __name__ == "__main__":
    main()
