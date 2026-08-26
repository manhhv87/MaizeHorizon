#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-tier AP (near/mid/far/all) on the hand-labelled test set.

Unlike eval_testset.py, this integrates the full precision-recall curve, so it does not depend
on a single operating point. Detections are ranked by confidence, as AP requires; eval_testset.py
ranks by IoU instead, because it scores one fixed operating point. The two are different procedures
and neither is a bug, but they are not interchangeable.

Size stratification: ground truth in the target tier is scored, and a prediction is neutralised if
it matches ground truth of another tier or falls in an `ignore` region. What happens to the rest is
a CHOICE, exposed as `fp_scale`:

  fp_scale=None (mac dinh)  moi prediction con lai la false positive cua tang dang xet.
  fp_scale=imgsz/max(W,H)   chi tinh la false positive khi CHINH prediction thuoc tang do.

Cai thu hai la quy uoc COCO. Cai thu nhat KHONG phai, du docstring nay tung ghi la
"COCO-style" -- do la sai sot va no quan trong: tren bo nay 88,6% prediction khong khop la
box co kich thuoc tang gan, nen o che do mac dinh, AP tang xa phan anh so box qua kho nhieu
hon la chat luong phat hien cay xa (0,020 so voi 0,357 tren nhanh stock).

Run at low confidence for a complete PR curve.

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


# COCO lay trung binh precision tai 101 muc recall cach deu, sau khi lam precision
# don dieu giam. Khac voi all-point cua VOC2010+, von tich phan dien tich chinh xac
# duoi duong PR. Hai cach cho so hoi khac nhau, nen phai chon mot va noi ro la cai nao.
REC_THRS = np.linspace(0.0, 1.0, 101)


def coco_ap(rec, prec):
    """AP noi suy 101 diem, dung nhu pycocotools.cocoeval."""
    if len(rec) == 0:
        return 0.0
    mpre = np.array(prec, dtype=float)
    for i in range(mpre.size - 1, 0, -1):          # precision don dieu giam
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # voi moi nguong recall, lay precision tai diem dau tien dat toi nguong do
    idx = np.searchsorted(rec, REC_THRS, side="left")
    q = np.where(idx < len(mpre), mpre[np.minimum(idx, len(mpre) - 1)], 0.0)
    q[idx >= len(mpre)] = 0.0
    return float(q.mean())


def ap_tier(per_image, tier, iou_thr, return_curve=False, fp_scale=None):
    """per_image[i] = (gts, igs, preds); gts=[(box,tier)], preds=[(conf,box)].

    fp_scale: neu dua vao (= imgsz/max(W,H)), mot prediction khong khop chi bi tinh
    la false positive cua tang dang xet khi CHINH NO co kich thuoc thuoc tang do,
    theo kieu COCO. Mac dinh None giu nguyen hanh vi cu de moi caller khong doi.

    Vi sao co tuy chon nay. Khong loc theo kich thuoc thi 88,6% false positive tinh
    vao AP tang xa lai la box co kich thuoc tang GAN (do tren nhanh +Mint, seed 0):
    AP tang xa khi do phan anh box lon doan sai nhieu hon la chat luong phat hien cay
    xa. Do la van de vi phep dao dau cat o duoc do bang chinh dai luong nay.

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
        # 4) COCO: detection nam ngoai dai kich thuoc cua tang bi danh ignore chu
        #    khong tinh la FP. Thieu buoc nay thi AP tang xa dem ca box qua kho --
        #    tren bo nay 88,6% prediction khong khop la box co kich thuoc tang gan.
        if fp_scale is not None and tier != "all":
            ph = (pb[3] - pb[1]) * fp_scale
            pt = "near" if ph >= 32 else ("mid" if ph >= 16 else "far")
            if pt != tier:
                continue
        fp[k] = 1
    tpc = np.cumsum(tp); fpc = np.cumsum(fp)
    rec = tpc / n_target
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    if return_curve:
        return coco_ap(rec, prec), n_target, (rec, prec, np.array([c for c, _, _ in all_preds]))
    return coco_ap(rec, prec), n_target


def predictions_for(model_path, items, args, diag=None):
    """Run one checkpoint -> per_image: (gts_with_tier, ignores, preds).

    `diag`: neu dua vao mot list, moi khung se them mot ban ghi
    (arm, frame, tile_idx, n_all_classes, n_plant, max_det, hit_cap). `n_all_classes`
    la so box Ultralytics tra ve TRUOC khi ta loc class plant -- do la con so duy nhat
    doi chieu duoc voi `max_det`, vi tran ay ap cho ca hai class.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    per_image = []
    for ip, w, h, plants, ignores in items:
        scale = args.imgsz / max(w, h)
        gts = [(b[:4], stratum(b, "pot", scale)) for b in plants]
        r = model.predict(ip, conf=args.conf, iou=0.6, imgsz=args.imgsz,
                          device=args.device, max_det=args.max_det, verbose=False)[0]
        preds = []
        n_all = 0
        if r.boxes is not None and len(r.boxes):
            cl = r.boxes.cls.cpu().numpy(); xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            n_all = len(cl)
            for j in range(len(cl)):
                if int(cl[j]) == args.plant_class:
                    preds.append((float(cf[j]), xy[j].tolist()))
        if diag is not None:
            diag.append(("full", os.path.basename(ip), -1, n_all, len(preds),
                         args.max_det, int(n_all >= args.max_det)))
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
    # Ti le POT: moi anh trong mot bo deu cung kich thuoc, nen mot gia tri la du.
    fp_sc = args.imgsz / max(items[0][1], items[0][2])
    for lbl, cps in groups:
        print(f"\n=== {lbl}  ({len(cps)} seed) ===")
        per_seed = {s: [] for s in STRATA}
        for cp in cps:
            per_image = predictions_for(cp, items, args)
            for s in STRATA:
                a, ng = ap_tier(per_image, s, args.iou, fp_scale=fp_sc)
                per_seed[s].append(a)
                if s == "all":
                    pass
            # GT counts are identical across seeds
            if ngt_ref is None:
                ngt_ref = {s: ap_tier(per_image, s, args.iou, fp_scale=fp_sc)[1] for s in STRATA}
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
