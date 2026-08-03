#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-tier (near/mid/far) recall and precision on the hand-labelled test set.

Takes --labels-dir (flat *.txt) and --images-dir (searched recursively by basename), so no
YOLO images/labels layout is required. Ignore-aware: only `plant` predictions are scored, and
an unmatched prediction whose centre falls in an `ignore` region is not counted as a false
positive. Ground-truth `plant` boxes are the recall denominator.

Aggregates over seeds (mean +/- sample std) and writes <out> plus <out>_curve.csv (recall vs POT).

  python eval_testset.py --labels-dir data/test/labels --images-dir data/test/images \
      --runs runs --tags stock nearfar --seeds 0 1 2 --iou 0.3 --imgsz 1280 --device 0
"""
import argparse
import csv
import glob
import os

import numpy as np

STRATA = ["near", "mid", "far", "all"]
# bin edges on box height in input pixels, for the recall-vs-POT curve
FINE_EDGES = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0, 96.0, float("inf")]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def sstd(a):
    a = np.asarray(a, float)
    n = int(np.sum(~np.isnan(a)))
    return float(np.nanstd(a, ddof=1)) if n > 1 else 0.0


def read_gt(lp, w, h, plant_cls=0, ignore_cls=1):
    """YOLO txt -> (plants, ignores); box = [x1,y1,x2,y2, base_y_norm, area_norm]."""
    plants, ignores = [], []
    if not os.path.exists(lp):
        return plants, ignores
    for line in open(lp, encoding="utf-8"):
        s = line.split()
        if len(s) < 5:
            continue
        c = int(float(s[0])); xc, yc, bw, bh = map(float, s[1:5])
        box = [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h,
               yc + bh / 2, bw * bh]
        if c == ignore_cls:
            ignores.append(box)
        elif c == plant_cls:
            plants.append(box)
    return plants, ignores


def iou_matrix(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)))
    g = np.array([b[:4] for b in gt], float); p = np.array(pred, float)
    x1 = np.maximum(g[:, None, 0], p[None, :, 0]); y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2]); y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]); ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    return inter / (ag[:, None] + ap[None, :] - inter + 1e-9)


def stratum(box, mode, scale):
    by, area = box[4], box[5]
    if mode == "ypos":
        return "near" if by >= 2 / 3 else ("mid" if by >= 1 / 3 else "far")
    if mode == "pot":
        pot_px = (box[3] - box[1]) * scale
        return "near" if pot_px >= 32 else ("mid" if pot_px >= 16 else "far")
    return "near" if area >= 0.02 else ("mid" if area >= 0.005 else "far")


def in_any(cx, cy, regions):
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in regions)


def recall_one(model_path, items, args):
    from ultralytics import YOLO
    model = YOLO(model_path)
    det = {s: 0 for s in STRATA}; tot = {s: 0 for s in STRATA}
    tp_t = {s: 0 for s in STRATA}; fp_t = {s: 0 for s in STRATA}
    nbins = len(FINE_EDGES) - 1
    cdet = np.zeros(nbins); ctot = np.zeros(nbins)
    for ip, w, h, gt, ig in items:
        scale = args.imgsz / max(w, h)
        r = model.predict(ip, conf=args.conf, iou=0.6, imgsz=args.imgsz, device=args.device,
                          max_det=args.max_det, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            cl = r.boxes.cls.cpu().numpy(); xy = r.boxes.xyxy.cpu().numpy()
            pred = xy[cl == args.plant_class]
        else:
            pred = np.zeros((0, 4))
        M = iou_matrix(gt, pred)
        gt_hit = {}
        if M.size:
            pairs = sorted(((M[gi, j], gi, j) for gi in range(len(gt)) for j in range(len(pred))
                            if M[gi, j] >= args.iou), reverse=True)
            used_g, used_p = set(), set()
            for _, gi, j in pairs:
                if gi in used_g or j in used_p:
                    continue
                used_g.add(gi); used_p.add(j); gt_hit[gi] = j
        matched_pred = set(gt_hit.values())
        for gi, gb in enumerate(gt):
            s = stratum(gb, args.mode, scale); hit = gi in gt_hit
            for key in (s, "all"):
                tot[key] += 1; det[key] += int(hit)
            if hit:
                tp_t[s] += 1; tp_t["all"] += 1
            pot_px = (gb[3] - gb[1]) * scale
            bi = min(max(int(np.searchsorted(FINE_EDGES, pot_px, side="right") - 1), 0), nbins - 1)
            ctot[bi] += 1; cdet[bi] += int(hit)
        for j in range(len(pred)):
            if j in matched_pred:
                continue
            cx = (pred[j, 0] + pred[j, 2]) / 2; cy = (pred[j, 1] + pred[j, 3]) / 2
            if in_any(cx, cy, ig):
                continue
            ph = (pred[j, 3] - pred[j, 1]) * scale
            ps = "near" if ph >= 32 else ("mid" if ph >= 16 else "far")
            fp_t[ps] += 1; fp_t["all"] += 1
    rec = {s: (det[s] / tot[s] if tot[s] else float("nan")) for s in STRATA}
    prec = {s: (tp_t[s] / (tp_t[s] + fp_t[s]) if (tp_t[s] + fp_t[s]) else float("nan")) for s in STRATA}
    return rec, tot, prec, cdet, ctot


def build_image_index(images_dir):
    """Map basename stem -> image path, searching recursively."""
    idx = {}
    for root, _, files in os.walk(images_dir):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMG_EXTS:
                idx.setdefault(stem, os.path.join(root, fn))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True, help="directory of *.txt label files")
    ap.add_argument("--images-dir", required=True, help="image root, searched recursively by basename")
    ap.add_argument("--runs", default="runs", help="directory holding <tag>_s<seed>/weights/best.pt")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--mode", choices=["ypos", "area", "pot"], default="pot")
    ap.add_argument("--iou", type=float, default=0.3, help="0.3 for the far tier: a few-pixel box cannot reach 0.5")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-det", type=int, default=300,
                    help="300 = Ultralytics default, as used in the paper. The cap applies to both classes "
                         "combined, so at conf=0.001 it truncates far plants; use 1000 to match eval_ap.py.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--weight", default="best.pt", choices=["best.pt", "last.pt"])
    ap.add_argument("--out", default="testset_stratified.csv")
    args = ap.parse_args()

    import cv2

    # pair labels with images by basename
    img_idx = build_image_index(args.images_dir)
    label_files = sorted(glob.glob(os.path.join(args.labels_dir, "*.txt")))
    if not label_files:
        raise SystemExit(f"No *.txt found in {args.labels_dir}")
    items = []; miss_img = []
    per_clip = {}
    for lp in label_files:
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = img_idx.get(stem)
        if ip is None:
            miss_img.append(stem); continue
        im = cv2.imread(ip)
        if im is None:
            miss_img.append(stem); continue
        h, w = im.shape[:2]
        plants, ignores = read_gt(lp, w, h, args.plant_class, args.ignore_class)
        items.append((ip, w, h, plants, ignores))
        clip = stem.rsplit("_f", 1)[0]
        per_clip.setdefault(clip, [0, 0]); per_clip[clip][0] += 1; per_clip[clip][1] += len(plants)
    if not items:
        raise SystemExit("No label could be paired with an image (check --images-dir).")
    n_bg = sum(1 for it in items if not it[3])
    print(f"[i] {len(items)} images paired ({n_bg} background, 0 plants); missing: {len(miss_img)}; "
          f"mode={args.mode} iou={args.iou} conf={args.conf} imgsz={args.imgsz}")
    for c, (ni, npl) in sorted(per_clip.items()):
        print(f"    {c}: {ni} images, {npl} plant GT")
    if miss_img:
        print(f"    [!] {len(miss_img)} labels have no image, e.g. {miss_img[:3]}")

    # collect checkpoints per tag
    groups = []
    for tag in args.tags:
        cps = [os.path.join(args.runs, f"{tag}_s{s}", "weights", args.weight) for s in args.seeds]
        cps = [c for c in cps if os.path.exists(c)]
        if cps:
            groups.append((tag, cps))
        else:
            print(f"[skip] {tag}: no checkpoint under {args.runs}")
    if not groups:
        raise SystemExit("No tag has a checkpoint.")

    rows_csv = [["model", "stratum", "n_gt", "recall_mean", "recall_std",
                 "prec_mean", "prec_std", "n_seeds"]]
    nbins = len(FINE_EDGES) - 1
    curve_rows = [["model", "pot_lo_px", "pot_hi_px", "n_gt", "recall_mean", "recall_std", "n_seeds"]]
    summary = {}
    for lbl, cps in groups:
        print(f"\n=== {lbl}  ({len(cps)} seed) ===")
        per = {s: [] for s in STRATA}; prec_per = {s: [] for s in STRATA}
        tot_ref = None; ctot_ref = None; seed_rec_bins = []
        for cp in cps:
            rec, tot, prec, cdet, ctot = recall_one(cp, items, args)
            tot_ref = tot; ctot_ref = ctot
            with np.errstate(invalid="ignore", divide="ignore"):
                seed_rec_bins.append(np.where(ctot > 0, cdet / np.maximum(ctot, 1), np.nan))
            for s in STRATA:
                per[s].append(rec[s]); prec_per[s].append(prec[s])
        summary[lbl] = {}
        for s in STRATA:
            rm, rs = float(np.nanmean(per[s])), sstd(per[s])
            pm, ps = float(np.nanmean(prec_per[s])), sstd(prec_per[s])
            summary[lbl][s] = (rm, rs)
            rows_csv.append([lbl, s, tot_ref[s], round(rm, 4), round(rs, 4),
                             round(pm, 4), round(ps, 4), len(cps)])
            print(f"  {s:5} recall = {rm:.4f} +/- {rs:.4f}   precision = {pm:.4f} +/- {ps:.4f}  (n_gt={tot_ref[s]})")
        R = np.array(seed_rec_bins)
        bmean = np.nanmean(R, axis=0); bstd = np.nanstd(R, axis=0)
        for b in range(nbins):
            lo, hi = FINE_EDGES[b], FINE_EDGES[b + 1]
            hi_s = "inf" if hi == float("inf") else f"{hi:.0f}"
            ng = int(ctot_ref[b]) if ctot_ref is not None else 0
            mv, sv = bmean[b], bstd[b]
            curve_rows.append([lbl, f"{lo:.0f}", hi_s, ng,
                               ("" if np.isnan(mv) else round(float(mv), 4)),
                               ("" if np.isnan(sv) else round(float(sv), 4)), len(cps)])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_csv)
    curve_path = os.path.splitext(args.out)[0] + "_curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(curve_rows)
    print(f"\n-> {args.out}\n-> {curve_path}")

    # paired recall deltas
    def show(a, b, why):
        if a in summary and b in summary:
            d = "  ".join(f"{s}:{summary[a][s][0] - summary[b][s][0]:+.4f}" for s in STRATA)
            print(f"  {a:16} - {b:16} {d}   [{why}]")
    print("\n=== recall delta by tier (near/mid/far/all) ===")
    print("# stage 1 (minting):")
    show("nearonly", "stock", "extra near data")
    show("nearfar", "nearonly", "far labels from minting")
    print("# stage 2 (distillation):")
    show("distill", "nearfar", "distill; confounded by aug being off")
    show("distill", "distill_shuffle", "control: >0 only if correct pairing helps")
    show("distill", "distill_synth", "control: >0 only if a real near view beats a synthetic one")


if __name__ == "__main__":
    main()
