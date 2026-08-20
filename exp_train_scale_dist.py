#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phan bo ti le cua TAP HUAN LUYEN — bao nhieu giam sat thuc su ton tai o tang xa.

Phan bien M4: bai ket luan "tang input khong mo rong duoc tam" va quy cho CAM BIEN.
Nhung neu tap huan luyen gan nhu khong co muc tieu tang xa thi ket luan do noi ve
GIAM SAT chu khong noi ve cam bien. Bai khong bao cao phan bo ti le cua tap train,
nen phan bien nay hien khong tra loi duoc.

Script dem, cho tung nhanh, so hop plant theo tang POT (tai imgsz trien khai) va theo
chieu cao hop goc, roi doi chieu voi tap test. Neu ti le tang xa trong train xap xi
trong test thi lap luan ve cam bien dung vung; neu train gan nhu khong co tang xa thi
phai ha giong ket luan.

  "$PY" exp_train_scale_dist.py --arm stock=data/arms/stock/train.txt \\
      --arm nearfar=data/arms/nearfar/train.txt --test data/test/labels \\
      --imgsz 1280 --out results/rebuttal/M4_train_scale.csv
"""
import argparse
import csv
import glob
import os

import numpy as np

from eval_testset import read_gt

TIERS = ["near", "mid", "far"]


def tier_of(pot):
    return "near" if pot >= 32 else ("mid" if pot >= 16 else "far")


def count_labels(label_paths, W, H, imgsz, plant_cls, ignore_cls):
    scale = imgsz / max(W, H)
    n = {t: 0 for t in TIERS}
    sub12 = 0
    heights = []
    for lp in label_paths:
        for b in read_gt(lp, W, H, plant_cls, ignore_cls)[0]:
            hb = b[3] - b[1]
            pot = hb * scale
            n[tier_of(pot)] += 1
            if pot < 12:
                sub12 += 1
            heights.append(hb)
    return n, sub12, np.array(heights)


def labels_from_list(list_file):
    """train.txt liet ke duong dan anh -> suy ra duong dan nhan."""
    out = []
    root = os.path.dirname(os.path.abspath(list_file))
    for ln in open(list_file, encoding="utf-8"):
        p = ln.strip()
        if not p:
            continue
        lp = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        for cand in (lp, os.path.join("data", lp), os.path.join(root, os.path.basename(lp))):
            if os.path.exists(cand):
                out.append(cand); break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], help="NHAN=duong/dan/train.txt")
    ap.add_argument("--test", help="thu muc nhan cua tap test, de doi chieu")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--frame-w", type=int, default=1920)
    ap.add_argument("--frame-h", type=int, default=1080)
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()

    jobs = [(s.partition("=")[0], labels_from_list(s.partition("=")[2])) for s in a.arm]
    if a.test:
        jobs.append(("test", sorted(glob.glob(os.path.join(a.test, "*.txt")))))

    rows = []
    print(f"{'tap':10} {'anh':>5} {'plant':>7} {'near':>7} {'mid':>7} {'far':>6} {'<12px':>6}  "
          f"{'%far':>6} {'h goc trung vi':>15}")
    for lab, lps in jobs:
        if not lps:
            print(f"{lab:10}  (khong doc duoc nhan)"); continue
        n, sub12, hs = count_labels(lps, a.frame_w, a.frame_h, a.imgsz, a.plant_class, a.ignore_class)
        tot = sum(n.values())
        pfar = 100.0 * n["far"] / max(tot, 1)
        print(f"{lab:10} {len(lps):>5} {tot:>7} {n['near']:>7} {n['mid']:>7} {n['far']:>6} "
              f"{sub12:>6}  {pfar:>5.1f}% {np.median(hs):>14.1f}")
        rows.append([lab, len(lps), tot, n["near"], n["mid"], n["far"], sub12,
                     f"{pfar:.2f}", f"{np.median(hs):.1f}"])

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "n_images", "n_plant", "near", "mid", "far", "sub12px",
                        "pct_far", "median_native_h_px"])
            w.writerows(rows)
        print(f"\n[OK] -> {a.out}")


if __name__ == "__main__":
    main()
