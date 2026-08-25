#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Duong precision-recall cho tang xa, mot duong moi nhanh.

Bai bao cao AP tang xa bang mot con so, nen doc gia khong thay AP do la hinh dang
gi: recall cao ma precision sup, hay ca hai deu thap. Hinh nay tra loi truc tiep,
va cho thay tran recall cua tang xa nam o dau.

  python plot_pr_far.py --labels-dir data/test/labels --images-dir data/test/images \\
      --runs runs --tags stock nearfar distill --seeds 0 1 2 --iou 0.3 \\
      --out paper/figures/fig_pr_far

Duong ve la trung binh theo seed tren mot luoi recall chung (noi suy), vung bong
la +-1 do lech chuan mau.
"""
import argparse
import os

import numpy as np

from eval_ap import ap_tier, predictions_for, build_image_index, read_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tags", nargs="+", default=["stock", "nearfar"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tier", default="far", choices=["near", "mid", "far", "all"])
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--weight", default="best.pt")
    ap.add_argument("--lang", choices=("en", "vi"), default="en")
    ap.add_argument("--out", default="fig_pr_far")
    a = ap.parse_args()

    L10N = {"en": dict(x="Recall", y="Precision",
                       t=f"Precision and recall on the {a.tier} tier (IoU {a.iou})"),
            "vi": dict(x="Recall", y="Precision",
                       t=f"Đường precision–recall ở tầng {'xa' if a.tier=='far' else a.tier} (IoU {a.iou})")}
    T = L10N[a.lang]

    import cv2, glob
    img_idx = build_image_index(a.images_dir)
    items = []
    for lp in sorted(glob.glob(os.path.join(a.labels_dir, "*.txt"))):
        ip = img_idx.get(os.path.splitext(os.path.basename(lp))[0])
        if ip is None:
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        plants, ignores = read_gt(lp, w, h, a.plant_class, a.ignore_class)
        items.append((ip, w, h, plants, ignores))
    if not items:
        raise SystemExit("khong ghep duoc nhan voi anh nao")
    print(f"[i] {len(items)} anh")
    grid = np.linspace(0.0, 1.0, 201)
    curves, aps = {}, {}
    for tag in a.tags:
        per_seed = []
        for sd in a.seeds:
            ckpt = os.path.join(a.runs, f"{tag}_s{sd}", "weights", a.weight)
            if not os.path.exists(ckpt):
                print(f"[skip] {ckpt}"); continue
            per_image = predictions_for(ckpt, items, a)
            val, n, (rec, prec, _) = ap_tier(per_image, a.tier, a.iou, return_curve=True,
                                             fp_scale=a.imgsz / 1920.0)
            # precision bao hoa (monotone) roi noi suy len luoi recall chung
            pm = np.maximum.accumulate(prec[::-1])[::-1]
            per_seed.append((np.interp(grid, rec, pm, left=pm[0] if len(pm) else 0.0, right=0.0), val))
            print(f"  {tag} s{sd}: AP={val:.4f}  n={n}")
        if per_seed:
            P = np.vstack([p for p, _ in per_seed])
            curves[tag] = (P.mean(0), P.std(0, ddof=1) if len(per_seed) > 1 else np.zeros_like(P[0]))
            aps[tag] = (float(np.mean([v for _, v in per_seed])),
                        float(np.std([v for _, v in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0)

    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType) cho CA HAI ngon ngu. Truoc day chi bat cho ban tieng Viet
    # (de giu dau), nen ban tieng Anh nhung Type 3 vao PDF -- nhieu nha xuat ban
    # tu choi Type 3, va chu trong hinh khong copy/tim kiem duoc.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    # Okabe--Ito, an toan voi mu mau
    COL = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
    STY = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    NICE = {"stock": "Stock", "nearfar": "+Mint", "distill": "+Distill",
            "distill_shuffle": "+Distill (shuffle)"}

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for i, tag in enumerate([t for t in a.tags if t in curves]):
        m, sd = curves[tag]
        lab = f"{NICE.get(tag, tag)}  AP={aps[tag][0]:.3f}"
        if aps[tag][1] > 0:
            lab += f"$\\pm${aps[tag][1]:.3f}"
        ax.plot(grid, m, STY[i % len(STY)], color=COL[i % len(COL)], lw=1.8, label=lab)
        ax.fill_between(grid, np.maximum(m - sd, 0), np.minimum(m + sd, 1),
                        color=COL[i % len(COL)], alpha=0.15, lw=0)
    ax.set_xlabel(T["x"]); ax.set_ylabel(T["y"]); ax.set_title(T["t"], fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200)
    print(f"[OK] -> {a.out}.pdf , {a.out}.png")


if __name__ == "__main__":
    main()
