#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-tier AP (near/mid/far/all) on the hand-labelled test set.

Unlike eval_testset.py, this integrates the full precision-recall curve, so it does not depend
on a single operating point. COCO-style size stratification: ground truth in the target tier is
scored, ground truth in other tiers and `ignore` regions neutralise a matching prediction, and
anything else unmatched is a false positive. Run at low confidence for a complete PR curve.

Reuses the matching helpers from eval_testset.py.

  python eval_ap.py --labels-dir data/test/labels --images-dir data/test/images \
      --runs runs --tags stock nearfar --seeds 0 1 2 --iou 0.3 --imgsz 1280 --device 0
"""
import argparse
import csv
import glob
import os

import numpy as np

from eval_testset import read_gt, build_image_index, in_any, stratum, sstd, IMG_EXTS

STRATA = ["near", "mid", "far", "all"]


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def voc_ap(rec, prec):
    """All-point AP (VOC2010+): area under PR after making precision monotone."""
    if len(rec) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_tier(per_image, tier, iou_thr, return_curve=False):
    """per_image[i] = (gts, igs, preds); gts=[(box,tier)], preds=[(conf,box)].

    return_curve=True tra them (recall, precision, conf) da sap theo conf giam dan,
    de ve duong PR. Mac dinh False nen moi caller cu khong doi.
    """
    all_preds = []  # (conf, img_idx, box)
    n_target = 0
    for i, (gts, igs, preds) in enumerate(per_image):
        for _, t in gts:
            if tier == "all" or t == tier:
                n_target += 1
        for c, pb in preds:
            all_preds.append((c, i, pb))
    if n_target == 0:
        return float("nan"), 0
    all_preds.sort(key=lambda x: -x[0])
    matched_t = [set() for _ in per_image]
    matched_o = [set() for _ in per_image]
    tp = np.zeros(len(all_preds)); fp = np.zeros(len(all_preds))
    for k, (c, i, pb) in enumerate(all_preds):
        gts, igs, _ = per_image[i]
        # 1) match target-tier GT first
        best, bj = iou_thr, -1
        for j, (gb, t) in enumerate(gts):
            if (tier != "all" and t != tier) or j in matched_t[i]:
                continue
            iou = box_iou(pb, gb)
            if iou >= best:
                best, bj = iou, j
        if bj >= 0:
            tp[k] = 1; matched_t[i].add(bj); continue
        # 2) other-tier GT neutralises the prediction
        if tier != "all":
            bo, boj = iou_thr, -1
            for j, (gb, t) in enumerate(gts):
                if t == tier or j in matched_o[i]:
                    continue
                iou = box_iou(pb, gb)
                if iou >= bo:
                    bo, boj = iou, j
            if boj >= 0:
                matched_o[i].add(boj); continue
        # 3) inside an ignore region: neutralise
        cx = (pb[0] + pb[2]) / 2; cy = (pb[1] + pb[3]) / 2
        if in_any(cx, cy, igs):
            continue
        fp[k] = 1
    tpc = np.cumsum(tp); fpc = np.cumsum(fp)
    rec = tpc / n_target
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    if return_curve:
        return voc_ap(rec, prec), n_target, (rec, prec, np.array([c for c, _, _ in all_preds]))
    return voc_ap(rec, prec), n_target


def predictions_for(model_path, items, args):
    """Run one checkpoint -> per_image: (gts_with_tier, ignores, preds)."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    per_image = []
    for ip, w, h, plants, ignores in items:
        scale = args.imgsz / max(w, h)
        gts = [(b[:4], stratum(b, "pot", scale)) for b in plants]
        r = model.predict(ip, conf=args.conf, iou=0.6, imgsz=args.imgsz,
                          device=args.device, max_det=args.max_det, verbose=False)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            cl = r.boxes.cls.cpu().numpy(); xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            for j in range(len(cl)):
                if int(cl[j]) == args.plant_class:
                    preds.append((float(cf[j]), xy[j].tolist()))
        per_image.append((gts, [b[:4] for b in ignores], preds))
    return per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.001, help="low, so the PR curve is complete")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--weight", default="best.pt", choices=["best.pt", "last.pt"])
    ap.add_argument("--out", default="testset_ap.csv")
    args = ap.parse_args()

    import cv2

    img_idx = build_image_index(args.images_dir)
    label_files = sorted(glob.glob(os.path.join(args.labels_dir, "*.txt")))
    if not label_files:
        raise SystemExit(f"No *.txt found in {args.labels_dir}")
    items = []
    for lp in label_files:
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
    print(f"[i] {len(items)} images; AP@IoU={args.iou} conf={args.conf} max_det={args.max_det} imgsz={args.imgsz}")

    groups = []
    for tag in args.tags:
        cps = [os.path.join(args.runs, f"{tag}_s{s}", "weights", args.weight) for s in args.seeds]
        cps = [c for c in cps if os.path.exists(c)]
        if cps:
            groups.append((tag, cps))
        else:
            print(f"[skip] {tag}: no checkpoint")
    if not groups:
        raise SystemExit("No tag has a checkpoint.")

    summary = {}
    rows = [["model", "stratum", "n_gt", "ap_mean", "ap_std", "n_seeds"]]
    ngt_ref = None
    for lbl, cps in groups:
        print(f"\n=== {lbl}  ({len(cps)} seed) ===")
        per_seed = {s: [] for s in STRATA}
        for cp in cps:
            per_image = predictions_for(cp, items, args)
            for s in STRATA:
                a, ng = ap_tier(per_image, s, args.iou)
                per_seed[s].append(a)
                if s == "all":
                    pass
            # GT counts are identical across seeds
            if ngt_ref is None:
                ngt_ref = {s: ap_tier(per_image, s, args.iou)[1] for s in STRATA}
        summary[lbl] = {}
        for s in STRATA:
            m, sd = float(np.nanmean(per_seed[s])), sstd(per_seed[s])
            summary[lbl][s] = m
            rows.append([lbl, s, ngt_ref[s], round(m, 4), round(sd, 4), len(cps)])
            print(f"  AP_{s:5} = {m:.4f} +/- {sd:.4f}   (n_gt={ngt_ref[s]})")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n-> {args.out}")

    def show(a, b, why):
        if a in summary and b in summary:
            d = "  ".join(f"{s}:{summary[a][s] - summary[b][s]:+.4f}" for s in STRATA)
            print(f"  {a:16} - {b:16} {d}   [{why}]")
    print("\n=== AP delta by tier (near/mid/far/all) ===")
    show("nearfar", "stock", "minting effect")
    show("distill", "nearfar", "distill vs +Mint; confounded by aug being off")
    show("distill", "distill_shuffle", "control: must be >0 if the pairing matters")
    print("\n# AP integrates the whole PR curve, so it does not depend on conf.")
    print("# Only tiers with large n_gt (near/mid) are reliable; far n_gt is small.")


if __name__ == "__main__":
    main()
