#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolution sweep: does detectability follow input pixels or sensor pixels?

Evaluates one fixed detector at several input sizes. For every ground-truth plant we record its
physical box height (native pixels, unchanged by imgsz), its pixels-on-target (POT = height *
imgsz/max(W,H)), and whether it was hit. Matching reuses eval_testset.py unchanged.

Writes three files:
  <prefix>_pot.csv    recall vs POT, one curve per imgsz
  <prefix>_phys.csv   recall vs physical height, one curve per imgsz (paired: same bins, same n)
  <prefix>_range.csv  h50, the physical height at 50% recall, per imgsz.
                      HAI cot: h50_phys_px khong loc bin thua, h50_phys_px_minn
                      da bo bin duoi --h50-min-n nhan. BAO CAO cot thu hai.
                      Tren bo 8K hai cot chenh 2.7 lan (26.67 vs 71.23 px) vi mot
                      bin 11 cay o dau duong cong keo diem cat di.
  <prefix>_prec.csv   per-tier recall and ignore-aware precision, per imgsz
  <prefix>_physprec.csv  recall and precision in the same fixed native bins, so a recall gain
                      at one input size can be checked against what it costs in precision

Eval-only keeps the model fixed and changes only the pixel budget. RUN_SCALING.md describes the
train-matched variant.

  python exp_resolution_sweep.py --labels-dir data/test/labels --images-dir data/test/images \
      --runs runs --tag nearfar --seeds 0 1 2 --imgsz-list 640 960 1280 1920 \
      --iou 0.3 --conf 0.25 --device 0 --out-prefix results/scaling/scaling
  python plot_scaling.py --prefix results/scaling/scaling
"""
import argparse
import glob
import os
import csv

import numpy as np

from eval_testset import (
    read_gt, build_image_index, iou_matrix, in_any, stratum, sstd, IMG_EXTS, FINE_EDGES,
)

# bin edges on physical box height (native px); fixed across imgsz
PHYS_EDGES = [0.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, float("inf")]
STRATA = ["near", "mid", "far", "all"]


def bin_index(edges, x):
    n = len(edges) - 1
    return min(max(int(np.searchsorted(edges, x, side="right") - 1), 0), n - 1)


def match_gt_hits(gt, pred, iou_thr, return_used_pred=False):
    """One-to-one greedy IoU -> set of hit GT indices; also the used prediction indices if asked."""
    M = iou_matrix(gt, pred)
    hits = set()
    used_g, used_p = set(), set()
    if M.size:
        pairs = sorted(((M[gi, j], gi, j) for gi in range(len(gt)) for j in range(len(pred))
                        if M[gi, j] >= iou_thr), reverse=True)
        for _, gi, j in pairs:
            if gi in used_g or j in used_p:
                continue
            used_g.add(gi); used_p.add(j); hits.add(gi)
    return (hits, used_p) if return_used_pred else hits


def predict_plant_boxes(model, ip, imgsz, args):
    r = model.predict(ip, conf=args.conf, iou=0.6, imgsz=imgsz,
                      device=args.device, max_det=args.max_det, verbose=False)[0]
    if r.boxes is not None and len(r.boxes):
        cl = r.boxes.cls.cpu().numpy(); xy = r.boxes.xyxy.cpu().numpy()
        return xy[cl == args.plant_class]
    return np.zeros((0, 4))


def eval_one_imgsz(cps, items, imgsz, args):
    """Return per-seed recall by POT bin and by physical bin, per-tier recall and precision, and n_gt.

    Precision is ignore-aware, as in eval_testset.recall_one."""
    # RT-DETR khong route qua YOLO() tran; dung loader chung cua rebuttal_common
    from rebuttal_common import load_detector
    npot = len(FINE_EDGES) - 1
    nphys = len(PHYS_EDGES) - 1
    pot_rec, phys_rec, tier_rec, tier_prec, phys_prec = [], [], [], [], []
    pot_n = np.zeros(npot); phys_n = np.zeros(nphys); tier_n = {s: 0 for s in STRATA}
    for si, cp in enumerate(cps):
        model = load_detector(cp, args.tag)
        pdet = np.zeros(npot); pth = np.zeros(nphys); pfp = np.zeros(nphys)
        tdet = {s: 0 for s in STRATA}
        tfp = {s: 0 for s in STRATA}
        for ip, w, h, gt, ig in items:
            scale = imgsz / max(w, h)
            pred = predict_plant_boxes(model, ip, imgsz, args)
            hits, used_p = match_gt_hits(gt, pred, args.iou, return_used_pred=True)
            for gi, gb in enumerate(gt):
                h_phys = gb[3] - gb[1]
                pot = h_phys * scale
                hit = int(gi in hits)
                bi = bin_index(FINE_EDGES, pot); pdet[bi] += hit
                bj = bin_index(PHYS_EDGES, h_phys); pth[bj] += hit
                s = stratum(gb, "pot", scale)
                tdet[s] += hit; tdet["all"] += hit
                if si == 0:
                    pot_n[bi] += 1; phys_n[bj] += 1
                    tier_n[s] += 1; tier_n["all"] += 1
            # ignore-aware FP: unmatched prediction whose centre is outside every ignore region
            for j in range(len(pred)):
                if j in used_p:
                    continue
                cx = (pred[j, 0] + pred[j, 2]) / 2; cy = (pred[j, 1] + pred[j, 3]) / 2
                if in_any(cx, cy, ig):
                    continue
                h_pred_native = pred[j, 3] - pred[j, 1]
                pfp[bin_index(PHYS_EDGES, h_pred_native)] += 1
                ph = h_pred_native * scale
                ps = "near" if ph >= 32 else ("mid" if ph >= 16 else "far")
                tfp[ps] += 1; tfp["all"] += 1
        # empty bin -> NaN, not 0.0; a fake 0.0 would distort the curve and h50
        pot_rec.append(np.where(pot_n > 0, pdet / np.maximum(pot_n, 1), np.nan))
        phys_rec.append(np.where(phys_n > 0, pth / np.maximum(phys_n, 1), np.nan))
        tier_rec.append({s: (tdet[s] / tier_n[s] if tier_n[s] else float("nan")) for s in STRATA})
        tier_prec.append({s: (tdet[s] / (tdet[s] + tfp[s]) if (tdet[s] + tfp[s]) else float("nan"))
                          for s in STRATA})
        # precision in the same fixed native bins as the recall curve, so the two are comparable
        # across input sizes. A true positive is binned by its ground-truth height, a false
        # positive by its own predicted height.
        denom = pth + pfp
        phys_prec.append(np.where(denom > 0, pth / np.maximum(denom, 1), np.nan))
    return (np.array(pot_rec), pot_n, np.array(phys_rec), phys_n, tier_rec, tier_n, tier_prec,
            np.array(phys_prec))


def h50(edges, rec_mean, n_gt=None, min_n=0):
    """Physical height at 50%% recall, linearly interpolated between bin midpoints.

    `min_n` bo cac bin co duoi ngan ay nhan truoc khi noi suy. KHONG dat mac dinh 0
    ma khong nghi: mot bin 11 cay o dau duong cong du de keo diem cat di gap ba lan.
    Da dinh tren bo 8K -- cung mot lo chay cho h50 = 26.67 px khi khong loc va
    71.23 px khi loc o min_n=100, tuc chenh 2.7 lan. Bao cao con so nao la mot lua
    chon phai noi ro, khong duoc de nguoi doc tinh co doc phai cot sai.
    """
    mids = [(edges[i] + (edges[i + 1] if np.isfinite(edges[i + 1]) else edges[i] + 32)) / 2
            for i in range(len(edges) - 1)]
    if n_gt is not None and min_n > 0:
        rec_mean = [r if (n_gt[i] or 0) >= min_n else float("nan")
                    for i, r in enumerate(rec_mean)]
    prev_m, prev_r = None, None
    for m, r in zip(mids, rec_mean):
        if np.isnan(r):
            continue
        if r >= 0.5:
            if prev_r is not None and prev_r < 0.5 and m != prev_m:
                t = (0.5 - prev_r) / (r - prev_r)
                return prev_m + t * (m - prev_m)
            return m
        prev_m, prev_r = m, r
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", required=True, help="one arm, e.g. nearfar or stock")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--imgsz-list", nargs="+", type=int, default=[640, 960, 1280, 1920])
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.25, help="0.25 = operating point; 0.001 = recall ceiling")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--weight", default="best.pt", choices=["best.pt", "last.pt"])
    ap.add_argument("--h50-min-n", type=int, default=100,
                    help="bo bin co duoi ngan ay nhan khi noi suy h50. Cot h50_phys_px "
                         "van la ban KHONG loc de doi chieu; cot h50_phys_px_minn moi la "
                         "ban dung bao cao.")
    ap.add_argument("--out-prefix", default="scaling")
    args = ap.parse_args()

    import cv2
    img_idx = build_image_index(args.images_dir)
    items = []
    for lp in sorted(glob.glob(os.path.join(args.labels_dir, "**", "*.txt"), recursive=True)):
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = img_idx.get(stem)
        if ip is None:
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        plants, ignores = read_gt(lp, w, h, args.plant_class, args.ignore_class)
        items.append((ip, w, h, plants, ignores))
    if not items:
        raise SystemExit("No label could be paired with an image.")

    cps = [os.path.join(args.runs, f"{args.tag}_s{s}", "weights", args.weight) for s in args.seeds]
    cps = [c for c in cps if os.path.exists(c)]
    if not cps:
        raise SystemExit(f"No checkpoint for tag={args.tag}")
    print(f"[i] {len(items)} images | tag={args.tag} | {len(cps)} seeds | imgsz={args.imgsz_list} "
          f"| IoU={args.iou} conf={args.conf}")

    pot_rows = [["imgsz", "pot_lo_px", "pot_hi_px", "n_gt", "recall_mean", "recall_std", "n_seeds"]]
    phys_rows = [["imgsz", "phys_lo_px", "phys_hi_px", "n_gt", "recall_mean", "recall_std", "n_seeds"]]
    range_rows = [["imgsz", "h50_phys_px", "h50_phys_px_minn", "min_n", "n_bins_used",
                   "recall_near", "recall_mid", "recall_far", "n_seeds"]]
    prec_rows = [["imgsz", "tier", "n_gt", "recall_mean", "recall_std",
                  "prec_mean", "prec_std", "n_seeds"]]
    physprec_rows = [["imgsz", "phys_lo_px", "phys_hi_px", "n_gt", "recall_mean", "recall_std",
                      "prec_mean", "prec_std", "n_seeds"]]

    for imgsz in args.imgsz_list:
        print(f"\n=== imgsz {imgsz} ===")
        (pot_rec, pot_n, phys_rec, phys_n,
         tier_rec, tier_n, tier_prec, phys_prec) = eval_one_imgsz(cps, items, imgsz, args)
        pm = np.nanmean(pot_rec, axis=0); ps = np.array([sstd(pot_rec[:, b]) for b in range(pot_rec.shape[1])])
        hm = np.nanmean(phys_rec, axis=0); hs = np.array([sstd(phys_rec[:, b]) for b in range(phys_rec.shape[1])])
        for b in range(len(FINE_EDGES) - 1):
            lo, hi = FINE_EDGES[b], FINE_EDGES[b + 1]
            hi_s = "inf" if not np.isfinite(hi) else f"{hi:.0f}"
            pot_rows.append([imgsz, f"{lo:.0f}", hi_s, int(pot_n[b]), round(float(pm[b]), 4),
                             round(float(ps[b]), 4), len(cps)])
        for b in range(len(PHYS_EDGES) - 1):
            lo, hi = PHYS_EDGES[b], PHYS_EDGES[b + 1]
            hi_s = "inf" if not np.isfinite(hi) else f"{hi:.0f}"
            phys_rows.append([imgsz, f"{lo:.0f}", hi_s, int(phys_n[b]), round(float(hm[b]), 4),
                              round(float(hs[b]), 4), len(cps)])
            qm = float(np.nanmean(phys_prec[:, b])) if np.any(np.isfinite(phys_prec[:, b])) else float("nan")
            qs = sstd([x for x in phys_prec[:, b] if np.isfinite(x)])
            physprec_rows.append([imgsz, f"{lo:.0f}", hi_s, int(phys_n[b]),
                                  round(float(hm[b]), 4), round(float(hs[b]), 4),
                                  round(qm, 4) if np.isfinite(qm) else "", round(qs, 4), len(cps)])
        tmean = {s: float(np.nanmean([tr[s] for tr in tier_rec])) for s in STRATA}
        pmean = {s: float(np.nanmean([tp[s] for tp in tier_prec])) for s in STRATA}
        h_50 = h50(PHYS_EDGES, hm)
        h_50f = h50(PHYS_EDGES, hm, phys_n, args.h50_min_n)
        n_used = int(sum(1 for x in phys_n if (x or 0) >= args.h50_min_n))
        range_rows.append([imgsz, round(h_50, 2) if np.isfinite(h_50) else "nan",
                           round(h_50f, 2) if np.isfinite(h_50f) else "nan",
                           args.h50_min_n, n_used,
                           round(tmean["near"], 4), round(tmean["mid"], 4), round(tmean["far"], 4), len(cps)])
        for s in STRATA:
            prec_rows.append([imgsz, s, tier_n[s],
                              round(tmean[s], 4), round(sstd([tr[s] for tr in tier_rec]), 4),
                              round(pmean[s], 4), round(sstd([tp[s] for tp in tier_prec]), 4), len(cps)])
        print(f"  recall near/mid/far = {tmean['near']:.3f}/{tmean['mid']:.3f}/{tmean['far']:.3f}"
              f"  | prec = {pmean['near']:.3f}/{pmean['mid']:.3f}/{pmean['far']:.3f}"
              f"  | h50_phys ~ {h_50:.1f}px  | n_gt = "
              f"{tier_n['near']}/{tier_n['mid']}/{tier_n['far']}")

    for name, rows in [("pot", pot_rows), ("phys", phys_rows), ("range", range_rows),
                       ("prec", prec_rows), ("physprec", physprec_rows)]:
        p = f"{args.out_prefix}_{name}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"-> {p}")
    print("\n# how to read:")
    print("#  _pot.csv   overlay recall-vs-POT per imgsz; curves that coincide mean POT governs detectability.")
    print("#  _phys.csv  overlay recall-vs-physical-height; a leftward shift means longer range.")
    print("#  _range.csv h50 per imgsz.")


if __name__ == "__main__":
    main()
