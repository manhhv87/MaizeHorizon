#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rut h50 (chieu cao hop GOC tai recall 0.5) tu mot lan quet do phan giai, theo tung imgsz.

Phan bien M3 hoi: ket luan ve san phat hien duoc rut ra tu YOLOv8-s, detector YEU NHAT
trong bon cai da thu (Bang 6: RT-DETR-l cho far recall 0.325 so voi 0.134). h50 chua he
duoc do tren RT-DETR-l. Script nay do, tren cung giao thuc, de tra loi hai cau rieng biet:

  1. h50 co PHANG theo imgsz khong?  -> neu co, phep tach sensor/input van dung
  2. MUC h50 co giong nhau giua cac kien truc khong? -> neu khong, san khong phai
     hang so cua cam bien ma phu thuoc detector

h50 noi suy tuyen tinh tren duong cong recall-theo-chieu-cao-goc, chi dung bin co
n_gt >= --min-n de khong doc nhieu tu bin thua.

  "$PY" exp_h50_by_arch.py --prefix results/rebuttal/M3_rtdetr_sweep --label rtdetr-l \\
      --out results/rebuttal/M3_h50_by_arch.csv
"""
import argparse
import csv
import collections

import numpy as np


def h50_from_phys(path, min_n):
    """{imgsz: (h50, n_bin dung duoc)} tu file *_phys.csv cua exp_resolution_sweep."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["recall_mean"] in ("", "nan", None):
            continue
        n = int(float(r.get("n_gt", 0) or 0))
        if n < min_n:
            continue
        lo = float(r["phys_lo_px"])
        hi = lo + 32.0 if str(r["phys_hi_px"]).lower().startswith("inf") else float(r["phys_hi_px"])
        by[int(r["imgsz"])].append(((lo + hi) / 2.0, float(r["recall_mean"]), n))
    out = {}
    for k, v in by.items():
        v.sort()
        x = [a for a, _, _ in v]; y = [b for _, b, _ in v]
        out[k] = (float(np.interp(0.5, y, x)) if min(y) <= 0.5 <= max(y) else float("nan"), len(v))
    return out


def h50_from_pot(path, model, min_n):
    """h50 tinh theo POT tu file *_curve.csv cua eval_testset (mot nhanh)."""
    v = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["model"] != model or r["recall_mean"] in ("", "nan", None):
            continue
        n = int(float(r.get("n_gt", 0) or 0))
        if n < min_n:
            continue
        lo = float(r["pot_lo_px"])
        hi = lo + 32.0 if str(r["pot_hi_px"]).lower().startswith("inf") else float(r["pot_hi_px"])
        v.append(((lo + hi) / 2.0, float(r["recall_mean"])))
    v.sort()
    x = [a for a, _ in v]; y = [b for _, b in v]
    return float(np.interp(0.5, y, x)) if min(y) <= 0.5 <= max(y) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pot-curve", action="append", default=[],
                    help="NHAN=duong/dan/*_curve.csv=MODEL -> h50 theo POT (che do M2)")
    ap.add_argument("--prefix", action="append", required=True,
                    help="tien to cua exp_resolution_sweep; lap lai cho tung kien truc")
    ap.add_argument("--label", action="append", required=True, help="nhan, cung thu tu voi --prefix")
    ap.add_argument("--min-n", type=int, default=100, help="bo bin thua nhan")
    ap.add_argument("--out")
    a = ap.parse_args()
    assert len(a.prefix) == len(a.label), "--prefix va --label phai cung so luong"

    if a.pot_curve:
        prows = []
        print(f"{'nhan':22} {'h50 POT (px)':>13}")
        for spec in a.pot_curve:
            lab, path, model = spec.split("=")
            h = h50_from_pot(path, model, 50)
            print(f"{lab:22} {h:>13.1f}")
            prows.append([lab, model, f"{h:.2f}"])
        if a.out:
            with open(a.out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["label", "model", "h50_pot_px"]); w.writerows(prows)
            print(f"\n[OK] -> {a.out}")
        return

    rows = []
    print(f"{'kien truc':12} {'imgsz':>6} {'h50 (px goc)':>13} {'n_bin':>6}")
    for pref, lab in zip(a.prefix, a.label):
        d = h50_from_phys(f"{pref}_phys.csv", a.min_n)
        for k in sorted(d):
            h, nb = d[k]
            print(f"{lab:12} {k:>6} {h:>13.1f} {nb:>6}")
            rows.append([lab, k, f"{h:.2f}", nb])
        vals = [d[k][0] for k in sorted(d) if k >= 960]
        if len(vals) > 1:
            print(f"{'':12} {'>=960':>6} bien do {max(vals)-min(vals):.1f} px "
                  f"(min {min(vals):.1f}, max {max(vals):.1f})")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arch", "imgsz", "h50_native_px", "n_bins_used"])
            w.writerows(rows)
        print(f"\n[OK] -> {a.out}")


if __name__ == "__main__":
    main()
