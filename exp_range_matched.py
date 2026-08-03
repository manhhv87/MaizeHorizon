#!/usr/bin/env python3
"""Same apparent size, different distance: does detectability follow pixels alone?

The headline claim is that resolved pixels govern far-field detectability. Native box
height is a monotone proxy for distance, however, so a single sweep cannot separate
pixel count from everything else that degrades with range (haze, defocus, motion blur,
plant-soil blending). Because the corpus spans a fivefold range of plant heights, a box
of a given native height arises at very different distances: a 52 px box is a 0.11 m
seedling at 3 m or a 0.57 m plant at 15 m.

This script recovers each annotated plant's distance from the imaging geometry, then
compares recall between near and far plants matched on native box height. Equal recall
supports the pixel account; worse recall at range means the half-recall height bundles
distance-dependent degradation.

Geometry, per clip: the horizon row fixes the tilt, the ground-contact row of each box
fixes its depression angle, and camera height converts that to distance.

    python exp_range_matched.py --labels-dir data/test/labels --images-dir data/test/images \
        --runs runs --tags stock nearfar --seeds 0 1 2 --out results/scaling/range_matched.csv

Matching is imported, never reimplemented.
"""

import argparse
import glob
import math
import os
import re
from collections import defaultdict

import cv2
import numpy as np

from eval_testset import read_gt, build_image_index
from exp_resolution_sweep import match_gt_hits

FP_DEFAULT = 1360.0          # focal length in pixels, from the C920 78 deg diagonal FOV
HCAM_DEFAULT = 0.65          # camera height above ground, m
BINS = [(24, 32), (32, 40), (40, 52), (52, 68), (68, 90), (90, 128)]   # native box height, px


def horizon_row(clip_dir, n=8):
    """Sky is far less saturated than soil, so the first saturated row is the horizon."""
    rs = []
    for f in sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))[:n]:
        im = cv2.imread(f)
        if im is None:
            continue
        sat = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1].mean(axis=1)
        top = sat[:int(im.shape[0] * 0.45)]
        rs.append(int(np.argmax(top > (top.min() + top.max()) / 2)))
    return float(np.median(rs)) if rs else None


def clip_of(path):
    return re.sub(r"_f\d+$", "", os.path.splitext(os.path.basename(path))[0])


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--frames-root", default="data/images")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tags", nargs="+", default=["stock", "nearfar"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001, help="ceiling, so thresholds are not a confound")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--fp", type=float, default=FP_DEFAULT)
    ap.add_argument("--hcam", type=float, default=HCAM_DEFAULT)
    ap.add_argument("--out", default="results/scaling/range_matched.csv")
    a = ap.parse_args()

    from ultralytics import YOLO

    idx = build_image_index(a.images_dir)
    labels = sorted(glob.glob(os.path.join(a.labels_dir, "*.txt")))

    # horizon per clip, so a re-mounted rig does not bias one clip's distances
    hor = {}
    for c in {clip_of(p) for p in labels}:
        for cand in (f"{a.frames_root}/{c}_test", f"{a.frames_root}/{c}"):
            if os.path.isdir(cand):
                hor[c] = horizon_row(cand)
                break
    print("horizon row per clip:", {k: (None if v is None else round(v)) for k, v in hor.items()})

    items = []
    for lp in labels:
        c = clip_of(lp)
        ip = idx.get(os.path.splitext(os.path.basename(lp))[0])
        if ip is None or hor.get(c) is None:
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        h_img, w_img = im.shape[:2]
        gt, ig = read_gt(lp, w_img, h_img, a.plant_class)
        items.append((ip, w_img, h_img, gt, ig, hor[c], c))
    print(f"{len(items)} frames, {sum(len(x[3]) for x in items)} plant boxes")

    # per (tag, seed): one row per GT box -> native height, distance, hit
    recs = defaultdict(list)
    for tag in a.tags:
        for s in a.seeds:
            wp = os.path.join(a.runs, f"{tag}_s{s}", "weights", "best.pt")
            if not os.path.exists(wp):
                print(f"  skip {tag} s{s}: no {wp}")
                continue
            model = YOLO(wp)
            for ip, w_img, h_img, gt, ig, yh, c in items:
                r = model.predict(ip, conf=a.conf, iou=0.6, imgsz=a.imgsz, device=a.device,
                                  max_det=a.max_det, verbose=False)[0]
                if r.boxes is not None and len(r.boxes):
                    cl = r.boxes.cls.cpu().numpy()
                    pred = r.boxes.xyxy.cpu().numpy()[cl == a.plant_class]
                else:
                    pred = np.zeros((0, 4))
                hits = match_gt_hits(gt, pred, a.iou)
                for gi, gb in enumerate(gt):
                    hpx = gb[3] - gb[1]
                    ybase = gb[3]
                    if ybase <= yh + 2:
                        continue
                    alpha = math.atan((ybase - yh) / a.fp)
                    d = a.hcam / math.tan(alpha)
                    if not (0.3 < d < 60):
                        continue
                    recs[(tag, s)].append((hpx, d, int(gi in hits), c))
            print(f"  {tag} s{s}: {len(recs[(tag, s)])} boxes scored")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    import csv
    with open(a.out, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["tag", "h_lo_px", "h_hi_px", "group", "n_gt", "recall_mean", "recall_std",
                       "d_median_m", "wilson_lo", "wilson_hi", "n_seeds"])

        for tag in a.tags:
            print(f"\n=== {tag} ===")
            print(f"  {'bin px':>10s} {'group':>5s} {'n':>6s} {'d med':>7s} {'recall':>16s}  {'95% CI':>14s}")
            for lo, hi in BINS:
                # split point on distance is taken once per bin, pooled over seeds
                pooled = [r for s in a.seeds for r in recs.get((tag, s), []) if lo <= r[0] < hi]
                if len(pooled) < 40:
                    continue
                dmed = float(np.median([r[1] for r in pooled]))
                out = {}
                for grp, keep in (("near", lambda d: d < dmed), ("far", lambda d: d >= dmed)):
                    per_seed, k_all, n_all, dl = [], 0, 0, []
                    for s in a.seeds:
                        sub = [r for r in recs.get((tag, s), []) if lo <= r[0] < hi and keep(r[1])]
                        if not sub:
                            continue
                        per_seed.append(sum(r[2] for r in sub) / len(sub))
                        k_all += sum(r[2] for r in sub); n_all += len(sub)
                        dl += [r[1] for r in sub]
                    if not per_seed:
                        continue
                    m = float(np.mean(per_seed))
                    sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
                    clo, chi = wilson(k_all, n_all)
                    out[grp] = (m, sd, n_all // max(len(per_seed), 1), float(np.median(dl)), clo, chi)
                    wcsv.writerow([tag, lo, hi, grp, n_all // max(len(per_seed), 1),
                                   f"{m:.4f}", f"{sd:.4f}", f"{np.median(dl):.2f}",
                                   f"{clo:.4f}", f"{chi:.4f}", len(per_seed)])
                    print(f"  {lo:4d}-{hi:<5d} {grp:>5s} {n_all//max(len(per_seed),1):>6d} "
                          f"{np.median(dl):>6.2f}m {m:>7.4f} +/- {sd:.4f}  [{clo:.3f},{chi:.3f}]")
                if "near" in out and "far" in out:
                    dm = out["near"][0] - out["far"][0]
                    overlap = not (out["near"][4] > out["far"][5] or out["far"][4] > out["near"][5])
                    print(f"  {'':10s} delta(near-far) = {dm:+.4f}   "
                          f"{'CI chong nhau' if overlap else 'CI TACH ROI'}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
