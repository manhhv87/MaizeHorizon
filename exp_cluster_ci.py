#!/usr/bin/env python3
"""Recall intervals clustered by plant, not by box.

The test set is 120 frames of continuous forward motion sampled a fifth of a second
apart, so one physical plant contributes many boxes. Treating those boxes as independent
Bernoulli trials makes every interval too narrow, and the far tier is where it matters:
82 boxes there are far fewer distinct plants.

This script attaches a plant identity to each annotated box using the human-verified
count ledger, reports how many distinct plants back each tier, and replaces the
box-level Wilson interval with a bootstrap that resamples plants (and, separately,
clips) as the unit of replication.

Recall is averaged over seeds, but the Wilson comparator is computed on the distinct
box count (n_box), not on the seed-pooled observation count (n_obs = n_box x n_seed):
the seeds re-score the same boxes, so pooling them is not extra sampling.

    python exp_cluster_ci.py --labels-dir data/test/labels --images-dir data/test/images \
        --runs runs --tags stock nearfar --seeds 0 1 2 --out results/detection/cluster_ci.csv

Matching is imported, never reimplemented.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

from eval_testset import read_gt, build_image_index, iou_matrix
from exp_resolution_sweep import match_gt_hits

TIERS = ["near", "mid", "far", "all"]


def clip_of(p):
    return re.sub(r"_f\d+$", "", os.path.splitext(os.path.basename(p))[0])


def tier_of(pot):
    return "near" if pot >= 32 else ("mid" if pot >= 16 else "far")


def load_ledgers(d):
    """clip -> {frame_slot: [(plant_id, box)]}, frame slots renumbered 0..39 in time order."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "ledger_*.json"))):
        if "_near" in f:
            continue
        led = json.load(open(f))
        clip = led["clip"].replace("_test", "")
        frames = sorted({b[0] for p in led["plants"] for b in p["boxes"]})
        slot = {fr: i for i, fr in enumerate(frames)}
        per = defaultdict(list)
        for p in led["plants"]:
            for b in p["boxes"]:
                per[slot[b[0]]].append((p["id"], np.array(b[1:5], dtype=float)))
        out[clip] = per
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def boot(groups, n_boot=4000, seed=0):
    """Resample whole groups with replacement; each group is a list of 0/1 outcomes."""
    rng = np.random.default_rng(seed)
    keys = list(groups)
    if not keys:
        return (float("nan"), float("nan"))
    est = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        hit = tot = 0
        for i in pick:
            g = groups[keys[i]]
            hit += sum(g); tot += len(g)
        if tot:
            est.append(hit / tot)
    return (float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))) if est else (float("nan"),) * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--ledger-dir", default="results/counting")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tags", nargs="+", default=["stock", "nearfar"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--link-iou", type=float, default=0.3, help="IoU to attach a ledger plant to a GT box")
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--out", default="results/detection/cluster_ci.csv")
    ap.add_argument("--plants-out", default="results/dataset/distinct_plants.csv",
                    help="bang box vs cay rieng biet; dat rong de bo qua")
    a = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    led = load_ledgers(a.ledger_dir)
    print("ledgers:", {k: sum(len(v) for v in p.values()) for k, p in led.items()})

    idx = build_image_index(a.images_dir)
    labels = sorted(glob.glob(os.path.join(a.labels_dir, "*.txt")))
    by_clip = defaultdict(list)
    for lp in labels:
        by_clip[clip_of(lp)].append(lp)

    # GT boxes with a plant id attached where the ledger covers them
    items = []
    linked = unlinked = 0
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
            pid = [None] * len(gt)
            if cand and len(gt):
                M = iou_matrix(gt, np.array([c[1] for c in cand]))
                pairs = sorted(((M[g, j], g, j) for g in range(len(gt)) for j in range(len(cand))
                                if M[g, j] >= a.link_iou), reverse=True)
                ug, up = set(), set()
                for _, g, j in pairs:
                    if g in ug or j in up:
                        continue
                    ug.add(g); up.add(j); pid[g] = f"{clip}#{cand[j][0]}"
            for i in range(len(gt)):
                if pid[i] is None:
                    pid[i] = f"{clip}#solo{slot}_{i}"   # unmatched box is its own cluster
                    unlinked += 1
                else:
                    linked += 1
            items.append((ip, w_img, h_img, gt, ig, pid, clip))
    print(f"GT boxes linked to a ledger plant: {linked}, unlinked (treated as singletons): {unlinked}")

    scale0 = a.imgsz / 1920.0
    # A2: distinct plants per tier, from the annotations alone
    tier_plants, tier_boxes = defaultdict(set), defaultdict(int)
    for _, w_img, h_img, gt, _, pid, _ in items:
        sc = a.imgsz / max(w_img, h_img)
        for i, gb in enumerate(gt):
            t = tier_of((gb[3] - gb[1]) * sc)
            for key in (t, "all"):
                tier_plants[key].add(pid[i]); tier_boxes[key] += 1
    print("\n=== A2: box vs cay rieng biet ===")
    print(f"  {'tang':6s} {'box':>6s} {'cay':>6s} {'box/cay':>8s}")
    prows = [["tier", "n_box", "n_plant", "boxes_per_plant"]]
    for t in TIERS:
        n_p = len(tier_plants[t]); n_b = tier_boxes[t]
        print(f"  {t:6s} {n_b:>6d} {n_p:>6d} {n_b/max(n_p,1):>8.2f}")
        prows.append([t, n_b, n_p, round(n_b / max(n_p, 1), 2)])
    # Dong "all" o day la so cay rieng biet TOAN BO, nho hon tong ba tang vi mot cay
    # doi tang giua cac khung. Bai trich dung con so nay, nen no phai nam trong mot CSV
    # chu khong chi in ra man hinh.
    if a.plants_out:
        os.makedirs(os.path.dirname(a.plants_out) or ".", exist_ok=True)
        with open(a.plants_out, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(prows)
        print(f"  -> {a.plants_out}")

    rows = []
    for tag in a.tags:
        # outcome per (seed, box); cluster keys are plant id and clip
        per_seed = {}
        for s in a.seeds:
            wp = os.path.join(a.runs, f"{tag}_s{s}", "weights", "best.pt")
            if not os.path.exists(wp):
                print(f"  skip {tag} s{s}")
                continue
            model = YOLO(wp)
            rec = []
            for ip, w_img, h_img, gt, ig, pid, clip in items:
                r = model.predict(ip, conf=a.conf, iou=0.6, imgsz=a.imgsz, device=a.device,
                                  max_det=a.max_det, verbose=False)[0]
                if r.boxes is not None and len(r.boxes):
                    cl = r.boxes.cls.cpu().numpy()
                    pred = r.boxes.xyxy.cpu().numpy()[cl == a.plant_class]
                else:
                    pred = np.zeros((0, 4))
                hits = match_gt_hits(gt, pred, a.iou)
                sc = a.imgsz / max(w_img, h_img)
                for i, gb in enumerate(gt):
                    t_i = tier_of((gb[3] - gb[1]) * sc)
                    hit = int(i in hits)
                    # Ghi ca ban ghi "all" song song, neu khong thi vong lap ben duoi
                    # loc theo r[0] == "all" se rong va dong tong bi bo qua im lang.
                    rec.append((t_i, pid[i], clip, hit))
                    rec.append(("all", pid[i], clip, hit))
            per_seed[s] = rec
            print(f"  {tag} s{s}: {len(rec)} boxes")

        if not per_seed:
            continue
        print(f"\n=== {tag} ===")
        print(f"  {'tang':6s} {'recall':>7s}  {'Wilson theo box':>22s}  {'bootstrap theo cay':>22s}  {'theo clip':>20s}")
        for t in TIERS:
            allrec = [r for s in per_seed for r in per_seed[s] if r[0] == t]
            if not allrec:
                continue
            n_obs = len(allrec)
            recall = sum(r[3] for r in allrec) / n_obs
            # The Wilson interval is the naive box-level comparator this script exists to
            # argue against, so it must use the number of DISTINCT annotated boxes. Every
            # seed scores the same boxes, so pooling the seeds counts each box n_seed
            # times and narrows the interval by sqrt(n_seed) -- which would understate the
            # comparator and overstate how much the clustering below widens it. Counting
            # one seed's records is exact even if a seed was skipped.
            first = next(iter(per_seed))
            n_box = sum(1 for r in per_seed[first] if r[0] == t)
            wl, wh = wilson(recall * n_box, n_box)
            # The bootstraps resample plants and clips, so their unit count is already
            # right; each group simply holds n_seed outcomes per box and the point
            # estimate it resamples is the seed-averaged recall.
            gp, gc = defaultdict(list), defaultdict(list)
            for _, p, c, h in allrec:
                gp[p].append(h); gc[c].append(h)
            pl, ph = boot(gp)
            cl_, ch = boot(gc)
            print(f"  {t:6s} {recall:>7.4f}  [{wl:.3f}, {wh:.3f}] w={wh-wl:.3f}  "
                  f"[{pl:.3f}, {ph:.3f}] w={ph-pl:.3f}  [{cl_:.3f}, {ch:.3f}]")
            rows.append([tag, t, n_box, len(per_seed), n_obs, len(gp), len(gc), f"{recall:.4f}",
                         f"{wl:.4f}", f"{wh:.4f}", f"{pl:.4f}", f"{ph:.4f}", f"{cl_:.4f}", f"{ch:.4f}"])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "tier", "n_box", "n_seed", "n_obs", "n_plant", "n_clip", "recall",
                    "wilson_lo", "wilson_hi", "plantboot_lo", "plantboot_hi",
                    "clipboot_lo", "clipboot_hi"])
        w.writerows(rows)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
