#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gom cac phep tuong phan h50 (M8) thanh mot bang de bai trich dan mot cho.

  python collect_h50_crossover.py --out results/rebuttal/M8_h50_crossover.csv

Bay phep tuong phan tra loi ba cau hoi khac nhau, va chi khi doc canh nhau thi chung
moi thanh mot lap luan:

  duoi muc goc     them diem anh dau vao co doi san khong  -> CO, tren ca hai nhanh
  tren muc goc     vuot cai cam bien da bat co them gi khong -> KHONG, tren ca ba nhanh
  doi ham mat mat  doi cach hoc co doi san khong           -> CO (nwd vs stock)

Cot `che_do` la thu duy nhat script nay them vao; moi con so deu chep nguyen tu file
nguon, khong tinh lai, de bang gop khong the lech voi file no gop.
"""
import argparse
import csv
import os

# (che_do, ten file) — thu tu o day la thu tu doc trong bai.
SRC = [
    ("duoi muc goc",   "M8_h50_stock_1920v960.csv"),
    ("duoi muc goc",   "M8_h50_nearfar_1920v960.csv"),
    ("tren muc goc",   "M8_h50_stock_2560v1920.csv"),
    ("tren muc goc",   "M8_h50_stock_3840v1920.csv"),
    ("tren muc goc",   "M8_h50_nwd_2560v1920.csv"),
    ("tren muc goc",   "M8_h50_nearfar_2560v1920.csv"),
    ("doi ham mat mat", "M8_h50_nwd_vs_stock.csv"),
]

COLS = ["conf", "tag_a", "imgsz_a", "tag_b", "imgsz_b", "n_seeds", "n_plant",
        "h50_a", "h50_b", "delta", "delta_lo", "delta_hi", "p_boot", "ket_luan"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="results/rebuttal")
    ap.add_argument("--out", default="results/rebuttal/M8_h50_crossover.csv")
    a = ap.parse_args()

    rows = [["che_do"] + COLS]
    missing = []
    for mode, fn in SRC:
        p = os.path.join(a.src_dir, fn)
        if not os.path.exists(p):
            missing.append(fn)
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rows.append([mode] + [r[c] for c in COLS])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"[OK] {len(rows) - 1} dong -> {a.out}")
    if missing:
        print(f"[!] thieu {len(missing)} file nguon: {', '.join(missing)}")
    print("\n  Tai TRAN (conf=0.001), noi h50 do kha nang phat hien gan nhu thuan tuy:")
    for r in rows[1:]:
        if r[1] == "0.001":
            print(f"    {r[0]:16s} {r[2]}@{r[3]:>4} vs {r[4]}@{r[5]:>4}: "
                  f"{r[10]:>6} [{r[11]}, {r[12]}]  p={r[13]:<6} {r[14]}")


if __name__ == "__main__":
    main()
