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

# Nhan theo ngon ngu, cho ban tieng Viet cua paper (paper/vi).
# Mac dinh 'en' nen hinh cua ban English khong doi.
L10N = {
    "en": {
           "titleA": "(A) Recall vs. pixels-on-target",
           "titleB": "(B) Recall vs. physical box height",
           "titleC": "(C) Half-recall height vs. input",
           "xlabelC": "Detector input width (px)",
           "ylabelC": "$h_{50}$ (px in original frame)",
           "native": "native",
           "armMain": "$+$Mint",
           "xlabelA": "Pixels-on-target (box height, px @ eval imgsz)",
           "xlabelB": "Physical box height (px in original frame); smaller = farther",
           "ylabel": "Per-plant recall"},
    "vi": {
           "titleA": "(A) Recall theo số điểm ảnh trên mục tiêu",
           "titleB": "(B) Recall theo chiều cao hộp gốc",
           "titleC": "(C) Chiều cao nửa recall theo đầu vào",
           "xlabelC": "Bề rộng đầu vào của detector (px)",
           "ylabelC": "$h_{50}$ (px trong khung gốc)",
           "native": "mức gốc",
           "armMain": "$+$Mint",
           "xlabelA": "Số điểm ảnh trên mục tiêu (chiều cao hộp, px @ imgsz đánh giá)",
           "xlabelB": "Chiều cao hộp trong khung gốc (px); nhỏ hơn = xa hơn",
           "ylabel": "Recall theo từng cây"},
}


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
    ap.add_argument("--lang", choices=("en", "vi"), default="en", help="ngon ngu nhan truc")
    ap.add_argument("--title", default=None, help="ghi de tieu de (mac dinh: theo --lang)")
    ap.add_argument("--h50-extra", nargs="*", default=[],
                    help="nhanh phu cho panel C, dang NHAN=file1.csv[,file2.csv]; "
                         "nhieu file thi gop lai (vd quet chinh + quet tren muc goc)")
    a = ap.parse_args()
    T = L10N[a.lang]
    out = a.out or f"{a.prefix}_fig"

    import matplotlib
    matplotlib.use("Agg")
    # Type 42 (TrueType) cho CA HAI ngon ngu. Truoc day chi bat cho ban tieng Viet
    # (de giu dau), nen ban tieng Anh nhung Type 3 vao PDF -- nhieu nha xuat ban
    # tu choi Type 3, va chu trong hinh khong copy/tim kiem duoc.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    pot = load(f"{a.prefix}_pot.csv")
    phys = load(f"{a.prefix}_phys.csv")
    # _range.csv chi co mot dong moi imgsz, khong dung `load` duoc
    import csv as _csv
    import os as _os
    def _read_h50(paths):
        out = {}
        for q in paths:
            if not _os.path.exists(q):
                continue
            for _r in _csv.DictReader(open(q, encoding="utf-8")):
                try:
                    out[int(_r["imgsz"])] = float(_r["h50_phys_px"])
                except (KeyError, ValueError):
                    pass
        return out

    h50s = _read_h50([f"{a.prefix}_range.csv"])
    series = [(T["armMain"], h50s, "#0072B2", "o")]
    _pal = ["#D55E00", "#009E73", "#CC79A7"]
    for _i, spec in enumerate(a.h50_extra):
        if "=" not in spec:
            print(f"[!] bo qua --h50-extra {spec!r}: thieu dau '='")
            continue
        _lab, _files = spec.split("=", 1)
        _d = _read_h50(_files.split(","))
        if _d:
            series.append((_lab, _d, _pal[_i % len(_pal)], "s^Dv"[_i % 4]))
        else:
            print(f"[!] --h50-extra {_lab!r}: khong doc duoc h50 nao")
    imgszs = sorted(set(pot) | set(phys))
    cmap = plt.get_cmap("viridis")
    colors = {im: cmap(i / max(1, len(imgszs) - 1)) for i, im in enumerate(imgszs)}
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    # Hai hang. Ba panel canh nhau o \linewidth chi con ~2,1 inch moi cai; hai hang
    # cho hang tren ~3,1 inch va panel C duoc tron be ngang.
    if h50s:
        fig = plt.figure(figsize=(11, 7.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.78], hspace=0.34, wspace=0.22)
        axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, :])
    else:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
        axC = None

    # --- Panel A: recall vs POT (collapse) ---
    axA.axvspan(0, 16, color="#f2c8c8", alpha=0.35, lw=0, zorder=0)
    axA.axvline(16, color="#c0392b", ls="--", lw=1.2, zorder=1)
    axA.text(18, 1.03, "T=16px", color="#c0392b", ha="left", fontsize=8)
    for i, im in enumerate(imgszs):
        if im not in pot:
            continue
        xs = [d[0] for d in pot[im]]; ys = [d[1] for d in pot[im]]; es = [d[2] for d in pot[im]]
        axA.plot(xs, ys, marker=markers[i % len(markers)], ms=4, lw=1.7, color=colors[im],
                 label=f"imgsz {im}", zorder=3)
        lo = [max(0, y - e) for y, e in zip(ys, es)]; up = [min(1, y + e) for y, e in zip(ys, es)]
        axA.fill_between(xs, lo, up, color=colors[im], alpha=0.12, zorder=2)
    axA.set_xlabel(T["xlabelA"])
    axA.set_ylabel(T["ylabel"])
    axA.set_xlim(0, 100); axA.set_ylim(-0.02, 1.08)
    axA.set_title(T["titleA"], fontsize=10)
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
    axB.axvline(52, color="#c0392b", ls="--", lw=1.0, zorder=1); axB.text(54, 1.03, "$h_{50}$ 52px", color="#c0392b", ha="left", fontsize=8)
    axB.set_xlabel(T["xlabelB"])
    axB.set_ylabel(T["ylabel"])
    axB.set_xlim(0, 160); axB.set_ylim(-0.02, 1.08)
    axB.set_title(T["titleB"], fontsize=10)
    axB.legend(loc="lower right", fontsize=8, framealpha=0.9); axB.grid(alpha=0.25)

    # --- Panel C: h50 theo dau vao. Day la cho DUY NHAT trong hinh nhin thay duoc
    # diem gay: h50 giam khi dau vao con khoi phuc diem anh cam bien da thu, cham
    # day tai muc goc, roi tang tro lai khi vuot qua.
    if axC is not None and h50s:
        allz = sorted({z for _, d, _, _ in series for z in d})
        for lab, d, col, mk in series:
            zs = sorted(d); ys = [d[z] for z in zs]
            axC.plot(zs, ys, marker=mk, ms=6, lw=1.8, color=col, label=lab, zorder=3)
            for z, y in zip(zs, ys):
                axC.annotate(f"{y:.1f}", (z, y), textcoords="offset points",
                             xytext=(0, 8 if lab == T["armMain"] else -13),
                             ha="center", fontsize=7.5, color=col)
            lo = min(ys); best = [z for z in zs if d[z] == lo][0]
            axC.scatter([best], [lo], s=110, facecolor="none", edgecolor=col,
                        lw=1.8, zorder=4)
        zs = allz
        axC.axvline(1920, color="#c0392b", ls="--", lw=1.1, zorder=1)
        axC.annotate(T["native"] + " 1920", (1920, axC.get_ylim()[1]),
                     textcoords="offset points", xytext=(5, -6), ha="left", va="top",
                     color="#c0392b", fontsize=8.5)
        if len(series) > 1:
            axC.legend(loc="upper center", fontsize=9, ncol=len(series), framealpha=0.9)
        # Log-scale van ve tick PHU co nhan, chong len nhan cua ta ("6x10^0" lan
        # vao giua "640"). Phai tat han bo dinh vi tick phu.
        import matplotlib.ticker as mticker
        axC.set_xscale("log")
        axC.xaxis.set_major_locator(mticker.FixedLocator(zs))
        axC.xaxis.set_major_formatter(mticker.FixedFormatter([str(z) for z in zs]))
        axC.xaxis.set_minor_locator(mticker.NullLocator())
        axC.tick_params(axis="x", labelsize=8)
        axC.set_xlabel(T["xlabelC"]); axC.set_ylabel(T["ylabelC"])
        axC.set_title(T["titleC"], fontsize=10); axC.grid(alpha=0.25)
        axC.margins(y=0.18)

    # KHONG dat tieu de mac dinh. Caption ngay duoi hinh da mo ta day du, va mot
    # cau trong ANH thi khong tang kiem nao doc duoc -- da dinh mot lan: tieu de cu
    # ghi "flat above input 960", than bai da sua tu lau ma cau trong anh van con,
    # va no chi lo ra khi co nguoi nhin hinh. Neu that su can thi dat bang --title.
    if a.title:
        fig.suptitle(a.title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98 if a.title else 1.0])
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"[OK] -> {out}.pdf , {out}.png")


if __name__ == "__main__":
    main()
