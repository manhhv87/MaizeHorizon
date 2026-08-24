#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do chi phi tinh toan theo kich thuoc dau vao, va quy ra tam hoat dong tren moi fps.

Bai khuyen nghi trien khai o input 2560 va xep hang input-scaling tren tiling dua
tren chi phi, nhung khong do chi phi lan nao. Script nay vá đúng chỗ đó: với mỗi
input size, đo latency (trung bình, p50, p95), throughput, bộ nhớ GPU đỉnh, rồi
ghép với h50 đã đo để ra tầm hoạt động ứng với mỗi mức fps.

Khong can du lieu moi: dung dung anh test da co.

  python exp_compute_cost.py --images-dir data/test/images --runs runs --tag nearfar \
      --seed 0 --sizes 640 960 1280 1920 2560 3840 --out results/rebuttal/M9_compute.csv

Luu y ve pham vi: day la mot GPU may tram, KHONG phai thiet bi edge. Con so tuyet
doi khong chuyen sang Jetson duoc; cai chuyen duoc la TI LE giua cac input size,
va do la thu quyet dinh danh doi trong bai.
"""
import argparse
import csv
import glob
import os
import statistics as st
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="nearfar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 960, 1280, 1920, 2560, 3840])
    ap.add_argument("--n-frames", type=int, default=40, help="so khung do moi input")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=3000)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--out", default="results/rebuttal/M9_compute.csv")
    a = ap.parse_args()

    import torch
    import cv2
    from ultralytics import YOLO

    ckpt = os.path.join(a.runs, f"{a.tag}_s{a.seed}", "weights", "best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"khong thay checkpoint: {ckpt}")
    frames = sorted(glob.glob(os.path.join(a.images_dir, "*.jpg")) +
                    glob.glob(os.path.join(a.images_dir, "*.png")))[:a.n_frames]
    if not frames:
        raise SystemExit(f"khong co anh trong {a.images_dir}")
    imgs = [cv2.imread(f) for f in frames]
    print(f"[i] {ckpt} | {len(imgs)} khung | GPU {torch.cuda.get_device_name(0)}")

    rows = []
    for sz in a.sizes:
        model = YOLO(ckpt)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for im in imgs[:a.warmup]:                       # warm-up: bo qua, chua tinh gio
            model.predict(im, imgsz=sz, conf=a.conf, max_det=a.max_det,
                          device=a.device, verbose=False)
        torch.cuda.synchronize()
        lat = []
        for im in imgs:
            t0 = time.perf_counter()
            model.predict(im, imgsz=sz, conf=a.conf, max_det=a.max_det,
                          device=a.device, verbose=False)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat.sort()
        peak = torch.cuda.max_memory_allocated() / 2**20
        row = dict(imgsz=sz,
                   lat_mean_ms=round(st.mean(lat), 1),
                   lat_p50_ms=round(lat[len(lat) // 2], 1),
                   lat_p95_ms=round(lat[int(0.95 * (len(lat) - 1))], 1),
                   fps=round(1000.0 / st.mean(lat), 1),
                   peak_mem_MiB=round(peak, 0),
                   n_frames=len(lat))
        rows.append(row)
        print(f"  imgsz {sz:>4}: {row['lat_mean_ms']:>7.1f} ms  p95 {row['lat_p95_ms']:>7.1f}  "
              f"{row['fps']:>6.1f} fps  {row['peak_mem_MiB']:>6.0f} MiB")
        del model

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[OK] -> {a.out}")

    base = next(r for r in rows if r["imgsz"] == 1280) if any(r["imgsz"] == 1280 for r in rows) else rows[0]
    print(f"\n  chi phi tuong doi so voi input {base['imgsz']}:")
    for r in rows:
        print(f"    {r['imgsz']:>4}: {r['lat_mean_ms']/base['lat_mean_ms']:.2f}x thoi gian, "
              f"{r['peak_mem_MiB']/base['peak_mem_MiB']:.2f}x bo nho")


if __name__ == "__main__":
    main()
