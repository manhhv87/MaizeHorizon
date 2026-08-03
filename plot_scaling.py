#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-panel figure from exp_resolution_sweep.py.

  (A) recall vs POT, one curve per imgsz
  (B) recall vs physical box height, one curve per imgsz

Curves that shift in (A) but collapse in (B) are the sensor-versus-input dissociation.

  python plot_scaling.py --prefix results/scaling/scaling --out paper/figures/fig_scaling
"""
import argparse
import csv
from collections import defaultdict


def load(path):
    d = defaultdict(list)  # imgsz -> [(x_mid, rec, std, n)]
    lo_key = None
    for r in csv.DictReader(open(path, encoding="utf-8")):
        lo_key = "pot_lo_px" if "pot_lo_px" in r else "phys_lo_px"
        hi_key = "pot_hi_px" if "pot_hi_px" in r else "phys_hi_px"
        lo = float(r[lo_key]); hi_s = r[hi_key]
        hi = lo + 32.0 if hi_s.lower().startswith("inf") else float(hi_s)
        xmid = (lo + hi) / 2.0
        rec = r["recall_mean"]

        def _to_int(v):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return -1   # n_gt khong hop le (vd '?') -> giu dong, KHONG crash
        ng = _to_int(r.get("n_gt"))
        if rec in ("", "nan", "nan\n", None) or ng == 0:  # bo bin rong (fix 1.2)
            continue
        d[int(r["imgsz"])].append((xmid, float(rec),
                                   float(r["recall_std"] or 0.0), max(ng, 0)))
    for k in d:
        d[k].sort()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="prefix written by exp_resolution_sweep (_pot.csv, _phys.csv)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Detection is governed by physical sensor resolution, not network input size")
    a = ap.parse_args()
    out = a.out or f"{a.prefix}_fig"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pot = load(f"{a.prefix}_pot.csv")
    phys = load(f"{a.prefix}_phys.csv")
    imgszs = sorted(set(pot) | set(phys))
    cmap = plt.get_cmap("viridis")
    colors = {im: cmap(i / max(1, len(imgszs) - 1)) for i, im in enumerate(imgszs)}
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Panel A: recall vs POT (collapse) ---
    axA.axvspan(0, 16, color="#f2c8c8", alpha=0.35, lw=0, zorder=0)
    axA.axvline(16, color="#c0392b", ls="--", lw=1.2, zorder=1)
    axA.text(16, 1.03, "T=16px", color="#c0392b", ha="center", fontsize=8)
    for i, im in enumerate(imgszs):
        if im not in pot:
            continue
        xs = [d[0] for d in pot[im]]; ys = [d[1] for d in pot[im]]; es = [d[2] for d in pot[im]]
        axA.plot(xs, ys, marker=markers[i % len(markers)], ms=4, lw=1.7, color=colors[im],
                 label=f"imgsz {im}", zorder=3)
        lo = [max(0, y - e) for y, e in zip(ys, es)]; up = [min(1, y + e) for y, e in zip(ys, es)]
        axA.fill_between(xs, lo, up, color=colors[im], alpha=0.12, zorder=2)
    axA.set_xlabel("Pixels-on-target (box height, px @ eval imgsz)")
    axA.set_ylabel("Per-plant recall")
    axA.set_xlim(0, 100); axA.set_ylim(-0.02, 1.08)
    axA.set_title("(A) Recall vs. pixels-on-target", fontsize=10)
    axA.legend(loc="lower right", fontsize=8, framealpha=0.9); axA.grid(alpha=0.25)

    # --- Panel B: recall vs physical box height (shift) ---
    for i, im in enumerate(imgszs):
        if im not in phys:
            continue
        xs = [d[0] for d in phys[im]]; ys = [d[1] for d in phys[im]]; es = [d[2] for d in phys[im]]
        axB.plot(xs, ys, marker=markers[i % len(markers)], ms=4, lw=1.7, color=colors[im],
                 label=f"imgsz {im}", zorder=3)
        lo = [max(0, y - e) for y, e in zip(ys, es)]; up = [min(1, y + e) for y, e in zip(ys, es)]
        axB.fill_between(xs, lo, up, color=colors[im], alpha=0.12, zorder=2)
    axB.axhline(0.5, color="#7f8c8d", ls=":", lw=1.0)
    axB.axvspan(0, 32, color="#f2c8c8", alpha=0.30, lw=0, zorder=0)   # floor vat ly ~24-32px
    axB.axvline(52, color="#c0392b", ls="--", lw=1.0, zorder=1); axB.text(52, 1.03, "h50~52px", color="#c0392b", ha="center", fontsize=8)
    axB.set_xlabel("Physical box height (px in original frame) -- smaller = farther")
    axB.set_ylabel("Per-plant recall")
    axB.set_xlim(0, 160); axB.set_ylim(-0.02, 1.08)
    axB.set_title("(B) Recall vs. physical box height", fontsize=10)
    axB.legend(loc="lower right", fontsize=8, framealpha=0.9); axB.grid(alpha=0.25)

    fig.suptitle(a.title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"[OK] -> {out}.pdf , {out}.png")


if __name__ == "__main__":
    main()
