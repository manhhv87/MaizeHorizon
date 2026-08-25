#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot recall versus pixels-on-target: the far-field cliff.

Reads a *_curve.csv written by eval_testset.py (model, pot_lo_px, pot_hi_px, n_gt, recall_mean,
recall_std, n_seeds) and draws recall per POT bin for each arm, shading the far tier. Writes .pdf
and .png.

  python plot_cliff.py --csv results/detection/testset_ceiling_curve.csv --out paper/figures/fig_cliff

Use the ceiling curve to expose the floor most clearly; conf 0.25 shows the operating point.
"""
import argparse
import csv
from collections import defaultdict

# Nhan theo ngon ngu. Ban tieng Viet cua paper (paper/vi) can hinh co nhan tieng Viet;
# mac dinh van la tieng Anh nen hinh cua ban English khong doi.
L10N = {
    "en": {"title": "Recall vs. pixels-on-target (forward-motion maize seedlings)",
           "far": "far (<16 px)", "mid": "mid", "near": "near (>=32 px)",
           "xlabel": "Pixels-on-target (box height, px @ imgsz 1280)",
           "ylabel": "Per-plant recall"},
    "vi": {"title": "Recall theo số điểm ảnh trên mục tiêu (cây ngô con, chuyển động tiến)",
           "far": "xa (<16 px)", "mid": "giữa", "near": "gần (≥ 32 px)",
           "xlabel": "Số điểm ảnh trên mục tiêu (chiều cao hộp, px @ imgsz 1280)",
           "ylabel": "Recall theo từng cây"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="*_curve.csv tu eval_testset/eval_stratified")
    ap.add_argument("--out", default="fig_cliff", help="tien to -> .pdf + .png")
    ap.add_argument("--lang", choices=("en", "vi"), default="en", help="ngon ngu nhan truc")
    ap.add_argument("--title", default=None, help="ghi de tieu de (mac dinh: theo --lang)")
    ap.add_argument("--only", nargs="+", default=None, help="plot only these arms (default: all)")
    a = ap.parse_args()
    T = L10N[a.lang]
    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType) cho CA HAI ngon ngu. Truoc day chi bat cho ban tieng Viet
    # (de giu dau), nen ban tieng Anh nhung Type 3 vao PDF -- nhieu nha xuat ban
    # tu choi Type 3, va chu trong hinh khong copy/tim kiem duoc.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    data = defaultdict(list)   # model -> [(x_mid, rec, std, n_gt)]
    for r in rows:
        m = r["model"]
        if a.only and m not in a.only:
            continue
        rec = r.get("recall_mean", "")
        if rec == "" or rec is None:
            continue
        lo = float(r["pot_lo_px"]); hi_s = r["pot_hi_px"]
        hi = lo + 32.0 if hi_s.lower().startswith("inf") else float(hi_s)
        xmid = (lo + hi) / 2.0
        std = float(r["recall_std"]) if r.get("recall_std", "") not in ("", None) else 0.0
        data[m].append((xmid, float(rec), std, int(float(r.get("n_gt", 0) or 0))))
    for m in data:
        data[m].sort()

    # display names
    NICE = {"stock": "Stock", "nearonly": "+Mint (near-only)", "nearfar": "+Mint",
            "distill": "+Distill", "distill_shuffle": "+Distill (shuffle)",
            "distill_synth": "+Mint+Distill (synth ctrl)"}
    ORDER = ["stock", "nearfar", "nearonly", "distill", "distill_shuffle", "distill_synth"]
    models = sorted(data, key=lambda m: ORDER.index(m) if m in ORDER else 99)

    plt.figure(figsize=(6.2, 4.0))
    # shade the far tier (<16px) and mark the floor
    plt.axvspan(0, 16, color="#f2c8c8", alpha=0.35, lw=0, zorder=0)
    plt.axvline(16, color="#c0392b", ls="--", lw=1.2, zorder=1)
    plt.axvline(32, color="#7f8c8d", ls=":", lw=1.0, zorder=1)
    plt.text(8, 1.02, T["far"], color="#c0392b", ha="center", fontsize=7)
    plt.text(24, 1.02, T["mid"], color="#555", ha="center", fontsize=8)
    plt.text(64, 1.02, T["near"], color="#555", ha="center", fontsize=8)

    markers = ["o", "s", "^", "D", "v", "P"]
    for i, m in enumerate(models):
        xs = [d[0] for d in data[m]]; ys = [d[1] for d in data[m]]; es = [d[2] for d in data[m]]
        plt.plot(xs, ys, marker=markers[i % len(markers)], ms=4, lw=1.8, label=NICE.get(m, m), zorder=3)
        lo = [max(0, y - e) for y, e in zip(ys, es)]; up = [min(1, y + e) for y, e in zip(ys, es)]
        plt.fill_between(xs, lo, up, alpha=0.15, zorder=2)

    plt.xlabel(T["xlabel"])
    plt.ylabel(T["ylabel"])
    plt.ylim(-0.02, 1.08); plt.xlim(0, max(110, max(d[0] for m in models for d in data[m]) + 5))
    plt.title(a.title or T["title"], fontsize=10)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.grid(alpha=0.25, zorder=0)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{a.out}.{ext}", dpi=200)
    print(f"[OK] -> {a.out}.pdf , {a.out}.png  | arms: {', '.join(models)}")


if __name__ == "__main__":
    main()
