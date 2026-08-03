#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-architecture control: is the far-tier floor specific to one detector?

Uses the paper protocol (POT-stratified, ignore-aware, one-to-one greedy IoU) but with a loader
that also handles RT-DETR, which plain YOLO() does not route.

Per tag it reports per-tier recall and precision at the operating point, far-tier recall at the
recall ceiling with a Wilson interval, and the recall-vs-POT curve. If no architecture clears
the floor threshold, the floor is not an artefact of one architecture.

  python exp_multiarch_eval.py --labels-dir data/test/labels --images-dir data/test/images \
      --runs runs --tags stock yolo11s yolov10s rtdetr-l --seeds 0 1 2 \
      --imgsz 1280 --iou 0.3 --device 0 --out results/baselines/rebuttal_multiarch.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuttal_common import (  # noqa: E402
    STRATA, FINE_EDGES, gather_items, greedy_hits, stratum, in_any, sstd, wilson,
    load_detector, predict_plant_boxes, find_checkpoints,
)


def eval_ckpt(cp, items, imgsz, conf, iou, device, plant_cls, arch_hint, max_det):
    """One checkpoint -> (recall, n, precision) per tier plus the POT curve."""
    model = load_detector(cp, arch_hint)
    det = {s: 0 for s in STRATA}; tot = {s: 0 for s in STRATA}
    tp = {s: 0 for s in STRATA}; fp = {s: 0 for s in STRATA}
    nb = len(FINE_EDGES) - 1
    cdet = np.zeros(nb); ctot = np.zeros(nb)
    far_tp_ceil = 0; far_n = 0
    for ip, w, h, gt, ig in items:
        scale = imgsz / max(w, h)
        pred, _ = predict_plant_boxes(model, ip, imgsz, conf, device, plant_cls, max_det)
        hit = greedy_hits(gt, pred, iou)
        matched = set(hit.values())
        # far recall ceiling: an extra low-confidence pass, counting far TPs only
        for gi, gb in enumerate(gt):
            s = stratum(gb, "pot", scale); h_ = gi in hit
            for k in (s, "all"):
                tot[k] += 1; det[k] += int(h_)
            if h_:
                tp[s] += 1; tp["all"] += 1
            pot = (gb[3] - gb[1]) * scale
            bi = min(max(int(np.searchsorted(FINE_EDGES, pot, side="right") - 1), 0), nb - 1)
            ctot[bi] += 1; cdet[bi] += int(h_)
        for j in range(len(pred)):
            if j in matched:
                continue
            cx = (pred[j, 0] + pred[j, 2]) / 2; cy = (pred[j, 1] + pred[j, 3]) / 2
            if in_any(cx, cy, ig):
                continue
            ph = (pred[j, 3] - pred[j, 1]) * scale
            ps = "near" if ph >= 32 else ("mid" if ph >= 16 else "far")
            fp[ps] += 1; fp["all"] += 1
    rec = {s: (det[s] / tot[s] if tot[s] else float("nan")) for s in STRATA}
    prec = {s: (tp[s] / (tp[s] + fp[s]) if (tp[s] + fp[s]) else float("nan")) for s in STRATA}
    return rec, tot, prec, cdet, ctot


def eval_ckpt_far_ceiling(cp, items, imgsz, iou, device, plant_cls, arch_hint, conf_ceiling, max_det):
    """Separate conf->0 pass: count far TP and far N for the Wilson interval."""
    model = load_detector(cp, arch_hint)
    far_tp = 0; far_n = 0
    for ip, w, h, gt, ig in items:
        scale = imgsz / max(w, h)
        far_idx = [gi for gi, gb in enumerate(gt) if (gb[3] - gb[1]) * scale < 16]
        far_n += len(far_idx)
        if not far_idx:
            continue
        pred, _ = predict_plant_boxes(model, ip, imgsz, conf_ceiling, device, plant_cls, max_det)
        hit = greedy_hits(gt, pred, iou)
        far_tp += sum(1 for gi in far_idx if gi in hit)
    return far_tp, far_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-ceiling", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--weight", default="best.pt")
    ap.add_argument("--max-det", type=int, default=300,
                    help="matches the paper (eval_testset uses the Ultralytics default of 300)")
    ap.add_argument("--floor-hi", type=float, default=0.30,
                    help="far@ceiling above this counts as clearing the floor")
    ap.add_argument("--out", default="rebuttal_multiarch.csv")
    a = ap.parse_args()

    items, per_clip, miss = gather_items(a.labels_dir, a.images_dir, a.plant_class, a.ignore_class)
    if not items:
        raise SystemExit("No label could be paired with an image; check --labels-dir/--images-dir.")
    print(f"[i] {len(items)} test images | imgsz={a.imgsz} iou={a.iou} conf={a.conf} "
          f"ceiling={a.conf_ceiling} max_det={a.max_det} | thieu anh: {len(miss)}")

    rows = [["arch", "stratum", "n_gt", "recall_mean", "recall_std", "prec_mean", "prec_std", "n_seeds"]]
    far_rows = [["arch", "far_recall_ceiling", "wilson_lo", "wilson_hi", "far_N", "n_seeds"]]
    summary = {}
    for tag in a.tags:
        cps = find_checkpoints(a.runs, tag, a.seeds, a.weight)
        if not cps:
            print(f"[skip] {tag}: no checkpoint under {a.runs}")
            continue
        print(f"\n=== {tag} ({len(cps)} seed) ===")
        per = {s: [] for s in STRATA}; prec_per = {s: [] for s in STRATA}; tot_ref = None
        far_tp_tot = 0; far_n_ref = 0
        for cp in cps:
            rec, tot, prec, cdet, ctot = eval_ckpt(cp, items, a.imgsz, a.conf, a.iou,
                                                   a.device, a.plant_class, tag, a.max_det)
            tot_ref = tot
            for s in STRATA:
                per[s].append(rec[s]); prec_per[s].append(prec[s])
            ftp, fn = eval_ckpt_far_ceiling(cp, items, a.imgsz, a.iou, a.device,
                                            a.plant_class, tag, a.conf_ceiling, a.max_det)
            far_tp_tot += ftp; far_n_ref = fn
        summary[tag] = {}
        for s in STRATA:
            rm, rs = float(np.nanmean(per[s])), sstd(per[s])
            pm, ps = float(np.nanmean(prec_per[s])), sstd(prec_per[s])
            summary[tag][s] = rm
            rows.append([tag, s, tot_ref[s], round(rm, 4), round(rs, 4), round(pm, 4), round(ps, 4), len(cps)])
            print(f"  {s:5} recall={rm:.4f}+/-{rs:.4f}  precision={pm:.4f}+/-{ps:.4f}  (n_gt={tot_ref[s]})")
        # point estimate = mean recall over seeds, but the Wilson interval uses the true n.
        # Pooling seeds over the same far set would be pseudo-replication and shrink the CI ~3x.
        mean_far = far_tp_tot / (far_n_ref * len(cps)) if far_n_ref else float("nan")
        p, lo, hi = wilson(int(round(mean_far * far_n_ref)), far_n_ref)
        far_rows.append([tag, round(p, 4), round(lo, 4), round(hi, 4), far_n_ref, len(cps)])
        print(f"  FAR recall @ceiling(conf={a.conf_ceiling}) = {p:.4f}  "
              f"Wilson95%=[{lo:.4f},{hi:.4f}]  (far N/seed={far_n_ref}, seeds={len(cps)})")

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    farp = os.path.splitext(a.out)[0] + "_far_ceiling.csv"
    with open(farp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(far_rows)
    print(f"\n-> {a.out}\n-> {farp}")

    # the floor holds if no architecture reaches a useful far recall; test max far@ceiling,
    # not overlapping CIs (a lower far recall is a stronger floor, not a weaker one)
    if len(far_rows) > 2:
        vals = [(r[0], r[1], r[2], r[3]) for r in far_rows[1:]]   # (arch, p, lo, hi)
        print("\n=== floor invariance ===")
        for name, p, lo, hi in vals:
            print(f"  {name:10} far@ceiling={p:.3f}  CI=[{lo:.3f},{hi:.3f}]")
        best = max(vals, key=lambda v: v[1]); max_far = best[1]
        print(f"  max far@ceiling = {max_far:.3f} ({best[0]})  |  nguong 'thoat san' floor_hi = {a.floor_hi}")
        if max_far < a.floor_hi:
            print(f"  -> floor holds: no architecture reaches useful far recall (best {best[0]}={max_far:.3f}). "
                  f"Dao dong giua arch = WITHIN-FLOOR + nhieu n. Ung ho C1.")
        else:
            print(f"  -> warning: {best[0]} far@ceiling {max_far:.3f} >= {a.floor_hi}, may clear the floor. "
                  f"Kiem near/mid cua no (co competent khong?) VA far-PRECISION (cao hay toan duong-gia?).")
        print("  (far@ceiling is an upper bound at conf->0; read it with far precision in the CSV: "
              "far cao MA precision thap = duong-gia, KHONG phai vuot san.)")


if __name__ == "__main__":
    main()
