#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 -- do recall theo KHOANG CACH MET that, roi so voi tam lam viec ma cong thuc
pinhole du doan.

Bai khang dinh `d_max = f*H_plant/(p*h50)` la tam lam viec. Nhung `d_max` duoc SUY RA
tu `h50`, nen cau "tai d_max thi recall = 0.5" dung theo DINH NGHIA, khong phai theo
phep do. Day la phep thu con thieu: tinh khoang cach cua tung cay tu hinh hoc da hieu
chuan, do recall theo khoang cach, xem cho cat 0.5 co roi dung cho cong thuc du doan.

Khoang cach lay tu HANG DAY cua box (giao thuc cua bai neo day box vao diem than-dat):

    alpha = tilt + atan((y_base - cy) / (f/p))      goc ha so voi phuong ngang
    d     = H_cam / tan(alpha)

Luu y ve tinh doc lap: `d` chi phu thuoc (f/p, tilt, H_cam), khong phu thuoc `h50`.
Nhung chieu cao box va y_base bi rang buoc boi quan he duong chan troi, nen day la
phep thu cho PHEP HOP THANH (hinh hoc + h50 -> tam), khong phai mot phep do hoan toan
doc lap. Phai khai ro nhu vay khi viet vao bai.

    "$PY" exp_range_validation.py --labels-dir ... --images-dir ... \\
        --tag stock --seeds 0 1 2 --imgsz 1280 --max-det 3000 \\
        --fp 5251 --tilt-deg 20.3 --cam-h 0.65 --h50-native 110.9 --plant-h 0.097 \\
        --out results/crosssite/hd_range_validation.csv
"""

import argparse
import glob
import os

import numpy as np

from eval_testset import build_image_index, iou_matrix, read_gt


def distances(gt, fp, tilt, cam_h, cy):
    """Khoang cach mat dat cua tung box, tu hang day."""
    y_base = gt[:, 3]
    alpha = tilt + np.arctan((y_base - cy) / fp)
    alpha = np.clip(alpha, 1e-4, np.pi / 2 - 1e-4)
    return cam_h / np.tan(alpha)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="stock")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=3000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--fp", type=float, required=True, help="tieu cu don vi pixel")
    ap.add_argument("--tilt-deg", type=float, required=True, help="goc nghieng duoi phuong ngang")
    ap.add_argument("--cam-h", type=float, default=0.65)
    ap.add_argument("--h50-native", type=float, required=True,
                    help="h50 tinh bang px native, de tinh d_max du doan")
    ap.add_argument("--plant-h", type=float, required=True, help="chieu cao cay trung vi, m")
    ap.add_argument("--edges", nargs="+", type=float,
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16])
    ap.add_argument("--out", default="range_validation.csv")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    idx = build_image_index(a.images_dir)
    items = []
    for lp in sorted(glob.glob(os.path.join(a.labels_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = idx.get(stem)
        if ip is None:
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        gt, ig = read_gt(lp, w, h, a.plant_class, a.ignore_class)
        if len(gt):
            items.append((ip, w, h, np.asarray(gt, dtype=float), ig))
    print(f"[i] {len(items)} anh, {sum(len(x[3]) for x in items)} box plant")

    tilt = np.radians(a.tilt_deg)
    edges = np.array(a.edges)
    nb = len(edges) - 1
    per_seed = []
    for sd in a.seeds:
        wpath = os.path.join(a.runs, f"{a.tag}_s{sd}", "weights", "best.pt")
        model = YOLO(wpath)
        det = np.zeros(nb)
        tot = np.zeros(nb)
        for ip, w, h, gt, ig in items:
            scale = a.imgsz / max(w, h)
            r = model.predict(ip, conf=a.conf, iou=0.6, imgsz=a.imgsz, device=a.device,
                              max_det=a.max_det, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                cl = r.boxes.cls.cpu().numpy()
                pred = r.boxes.xyxy.cpu().numpy()[cl == a.plant_class]
            else:
                pred = np.zeros((0, 4))
            M = iou_matrix(gt, pred)
            hit = np.zeros(len(gt), dtype=bool)
            if M.size:
                pairs = sorted(((M[gi, j], gi, j) for gi in range(len(gt))
                                for j in range(len(pred)) if M[gi, j] >= a.iou), reverse=True)
                ug, up = set(), set()
                for _, gi, j in pairs:
                    if gi in ug or j in up:
                        continue
                    ug.add(gi)
                    up.add(j)
                    hit[gi] = True
            d = distances(gt, a.fp, tilt, a.cam_h, h / 2)
            b = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, nb - 1)
            ok = (d >= edges[0]) & (d < edges[-1])
            for k in range(nb):
                m = ok & (b == k)
                tot[k] += m.sum()
                det[k] += hit[m].sum()
        per_seed.append(np.where(tot > 0, det / np.maximum(tot, 1), np.nan))
        print(f"  seed {sd}: xong")

    R = np.vstack(per_seed)
    mean = np.nanmean(R, axis=0)
    std = np.nanstd(R, axis=0, ddof=1) if len(a.seeds) > 1 else np.zeros(nb)
    mids = (edges[:-1] + edges[1:]) / 2

    d_pred = a.fp * a.plant_h / a.h50_native
    d_meas = None
    for i in range(nb - 1):
        y0, y1 = mean[i], mean[i + 1]
        if np.isfinite(y0) and np.isfinite(y1) and y0 >= 0.5 > y1:
            d_meas = mids[i] + (0.5 - y0) / (y1 - y0) * (mids[i + 1] - mids[i])
            break

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("d_lo_m,d_hi_m,n_gt,recall_mean,recall_std,n_seeds\n")
        for k in range(nb):
            f.write(f"{edges[k]},{edges[k+1]},{int(tot[k])},"
                    f"{mean[k]:.4f},{std[k]:.4f},{len(a.seeds)}\n")

    print(f"\n  {'d (m)':>10} {'n':>7} {'recall':>8}")
    for k in range(nb):
        if tot[k]:
            print(f"  {edges[k]:4.0f}-{edges[k+1]:<5.0f} {int(tot[k]):7d} {mean[k]:8.3f}")
    print(f"\n  d_max DU DOAN  (f/p * P / h50) = {d_pred:.2f} m")
    print(f"  d_max DO DUOC  (recall cat 0.5) = "
          + (f"{d_meas:.2f} m   lech {abs(d_meas-d_pred)/d_pred*100:.0f}%"
             if d_meas else "khong cat 0.5 trong dai da quet"))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
