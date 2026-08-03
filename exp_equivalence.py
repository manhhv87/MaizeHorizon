#!/usr/bin/env python3
"""Is recall above input 960 equivalent, or merely not-shown-to-differ?

The paper asserts a null: recall collapses onto one curve for inputs at or above 960.
A null needs an equivalence test, not a failed difference test. Seed-level comparison
at n=3 has almost no power, so this script exploits the paired design instead: the same
plants are evaluated at every input size, so each plant contributes a within-plant
difference and the comparison is paired at the level that matters.

Physical height bins are fixed in native pixels, so the same plants populate each bin at
every input size. Intervals resample plants, since boxes from one plant are not
independent. Rather than pick an equivalence margin that flatters the result, the script
reports the interval and the smallest margin the data would support.

    python exp_equivalence.py --labels-dir data/test/labels --images-dir data/test/images \
        --runs runs --tag nearfar --seeds 0 1 2 --sizes 640 960 1280 1920 \
        --out results/scaling/equivalence.csv

Matching and ledger linkage are imported, never reimplemented.
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from eval_testset import read_gt, build_image_index, iou_matrix
from exp_resolution_sweep import match_gt_hits
from exp_cluster_ci import load_ledgers, clip_of

PHYS_EDGES = [16, 24, 32, 48, 64, 96, 128, 192]


def phys_bin(h):
    for i in range(len(PHYS_EDGES) - 1):
        if PHYS_EDGES[i] <= h < PHYS_EDGES[i + 1]:
            return f"{PHYS_EDGES[i]}-{PHYS_EDGES[i+1]}"
    return f">={PHYS_EDGES[-1]}" if h >= PHYS_EDGES[-1] else None


def paired_boot(per_plant_a, per_plant_b, n_boot=4000, seed=0):
    """Bootstrap the recall difference, resampling plants so paired structure is kept."""
    keys = sorted(set(per_plant_a) & set(per_plant_b))
    if not keys:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        ha = ta = hb = tb = 0
        for i in pick:
            k = keys[i]
            ha += sum(per_plant_a[k]); ta += len(per_plant_a[k])
            hb += sum(per_plant_b[k]); tb += len(per_plant_b[k])
        if ta and tb:
            est.append(ha / ta - hb / tb)
    if not est:
        return (float("nan"),) * 3
    return (float(np.mean(est)), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--ledger-dir", default="results/counting")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="nearfar")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 960, 1280, 1920])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--link-iou", type=float, default=0.3)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.05, help="equivalence margin, recall units")
    ap.add_argument("--out", default="results/scaling/equivalence.csv")
    a = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    led = load_ledgers(a.ledger_dir)
    idx = build_image_index(a.images_dir)
    by_clip = defaultdict(list)
    for lp in sorted(glob.glob(os.path.join(a.labels_dir, "*.txt"))):
        by_clip[clip_of(lp)].append(lp)

    items = []
    for clip, lps in by_clip.items():
        for slot, lp in enumerate(sorted(lps)):
            ip = idx.get(os.path.splitext(os.path.basename(lp))[0])
            if ip is None:
                continue
            im = cv2.imread(ip)
            if im is None:
                continue
            h_img, w_img = im.shape[:2]
            gt, ig = read_gt(lp, w_img, h_img, a.plant_class)
            cand = led.get(clip, {}).get(slot, [])
            pid = [f"{clip}#solo{slot}_{i}" for i in range(len(gt))]
            if cand and len(gt):
                M = iou_matrix(gt, np.array([c[1] for c in cand]))
                pairs = sorted(((M[g, j], g, j) for g in range(len(gt)) for j in range(len(cand))
                                if M[g, j] >= a.link_iou), reverse=True)
                ug, up = set(), set()
                for _, g, j in pairs:
                    if g in ug or j in up:
                        continue
                    ug.add(g); up.add(j); pid[g] = f"{clip}#{cand[j][0]}"
            items.append((ip, w_img, h_img, gt, pid))
    print(f"{len(items)} frames, {sum(len(x[3]) for x in items)} plant boxes")

    # hits[(size, bin)][plant] = list of 0/1 pooled over seeds
    hits = defaultdict(lambda: defaultdict(list))
    n_bin = defaultdict(int)
    for s in a.seeds:
        wp = os.path.join(a.runs, f"{a.tag}_s{s}", "weights", "best.pt")
        if not os.path.exists(wp):
            print(f"  skip seed {s}")
            continue
        model = YOLO(wp)
        for size in a.sizes:
            for ip, w_img, h_img, gt, pid in items:
                r = model.predict(ip, conf=a.conf, iou=0.6, imgsz=size, device=a.device,
                                  max_det=a.max_det, verbose=False)[0]
                if r.boxes is not None and len(r.boxes):
                    cl = r.boxes.cls.cpu().numpy()
                    pred = r.boxes.xyxy.cpu().numpy()[cl == a.plant_class]
                else:
                    pred = np.zeros((0, 4))
                hit = match_gt_hits(gt, pred, a.iou)
                for i, gb in enumerate(gt):
                    b = phys_bin(gb[3] - gb[1])      # native height, identical across sizes
                    if b is None:
                        continue
                    hits[(size, b)][pid[i]].append(int(i in hit))
                    if s == a.seeds[0] and size == a.sizes[0]:
                        n_bin[b] += 1
            print(f"  {a.tag} s{s} @ {size}: done")

    bins = [b for b in sorted(n_bin, key=lambda x: int(x.split("-")[0].lstrip(">="))) if n_bin[b] >= 25]
    comps = [(x, y) for i, x in enumerate(a.sizes) for y in a.sizes[i + 1:] if x >= 960]

    rows = []
    print(f"\n=== paired recall difference, plants resampled, margin {a.margin:+.2f} ===")
    for lo, hi in comps:
        print(f"\n  {lo} vs {hi}")
        print(f"    {'bin (native px)':>16s} {'n box':>6s} {'n cay':>6s} {'delta':>8s} {'95% CI':>18s}  verdict")
        for b in bins:
            A, B = hits[(lo, b)], hits[(hi, b)]
            npl = len(set(A) & set(B))
            if npl < 15:
                continue
            d, cl, ch = paired_boot(A, B)
            if np.isnan(d):
                continue
            if cl > -a.margin and ch < a.margin:
                v = "TUONG DUONG"
            elif cl > 0 or ch < 0:
                v = "KHAC BIET"
            else:
                v = "khong ket luan"
            smallest = max(abs(cl), abs(ch))
            print(f"    {b:>16s} {n_bin[b]:>6d} {npl:>6d} {d:>+8.4f} [{cl:+.4f},{ch:+.4f}]  {v} (can margin >={smallest:.3f})")
            rows.append([a.tag, lo, hi, b, n_bin[b], npl, f"{d:.4f}", f"{cl:.4f}", f"{ch:.4f}",
                         f"{smallest:.4f}", v])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "size_a", "size_b", "phys_bin", "n_box", "n_plant",
                    "delta", "ci_lo", "ci_hi", "min_margin_for_equivalence", "verdict"])
        w.writerows(rows)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
