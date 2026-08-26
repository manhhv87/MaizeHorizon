#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h50 co phu thuoc vao nguong bin thua khong, va uoc luong khong can nguong.

  # (a) do nhay theo min_n, doc tu CSV da co, KHONG can GPU
  python exp_h50_robustness.py --from-csv results/crosssite/hd_sweep_phys.csv \\
      --out results/crosssite/hd_h50_minn_sensitivity.csv

  # (b) uoc luong bang hoi quy don dieu tren tung hop, co bootstrap (can GPU)
  python exp_h50_robustness.py --labels-dir <8K/labels> --images-dir <8K/images> \\
      --runs runs --tag stock --seeds 0 1 2 3 4 --imgsz-list 1280 1920 2560 3840 \\
      --max-det 3000 --device 0 --out results/crosssite/hd_h50_isotonic.csv

Van de. `exp_resolution_sweep.py` bo cac bin co duoi `--h50-min-n` (mac dinh 100)
nhan truoc khi noi suy. Nguong nay KHONG co trong phan Methods, duoc them vao sau
khi da thay sai lech tren bo 8K, va toan bo d50 cua bai phu thuoc vao no: o input
2560 thi h50 la 26,67 px neu khong loc va 71,23 px neu loc, chenh 2,7 lan.

Nguyen nhan that khong phai "bin thua thi nhieu" ma la duong cong recall KHONG DON
DIEU. O input 2560, bin 24--32 px co 11 hop va recall 0,600, trong khi bin 32--48 px
co 220 hop va recall 0,365. Quy tac "bin dau tien vuot 0,5" vi the bam vao 11 hop.
Mot nguong dem la cach vong tranh trieu chung; cach dung la ep tinh don dieu.

Script nay lam ca hai:

  minn   tinh lai h50 tren mot day nguong. Neu ket luan chi doi trong pham vi hep
         thi nguong khong phai cho chon cho vua ket qua; neu no nhay lung tung thi
         con so khong dung duoc.
  iso    hoi quy don dieu (pool-adjacent-violators) trong tiep tren cap (chieu cao,
         trung/truot) cua TUNG HOP, khong chia bin, khong nguong. Day la uoc luong
         hop ly cuc dai cua mot ham recall khong tang theo chieu cao giam, tuc dung
         gia thiet vat ly ma bai dang dua vao, va khong con tham so tuy chon nao.
         Bootstrap theo ANH (bo 8K khong co so cai cay nhu bo 1080p).
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from eval_testset import read_gt, build_image_index
from exp_resolution_sweep import PHYS_EDGES, h50
from eval_ap import box_iou


def sensitivity(phys_csv, grid):
    """Tinh lai h50 tu CSV da bin san, tren mot day min_n."""
    by = defaultdict(list)
    for r in csv.DictReader(open(phys_csv, encoding="utf-8")):
        by[int(r["imgsz"])].append(r)
    rows = [["imgsz", "min_n", "n_bins_used", "h50_px"]]
    for z in sorted(by):
        rs = sorted(by[z], key=lambda r: float(r["phys_lo_px"]))
        rec = [float(r["recall_mean"]) if r["recall_mean"] not in ("", "nan") else float("nan")
               for r in rs]
        n = [int(r["n_gt"]) for r in rs]
        for m in grid:
            v = h50(PHYS_EDGES, rec, n, m)
            rows.append([z, m, sum(1 for x in n if x >= m),
                         "" if not np.isfinite(v) else round(float(v), 2)])
    return rows


def pava(y, w):
    """Pool-adjacent-violators: chieu y (co trong so w) len ham KHONG TANG.

    Tra ve gia tri da lam tron don dieu. Dung truc tiep, khong qua sklearn, de
    script khong them phu thuoc va de doc duoc chinh xac no lam gi.
    """
    y = list(map(float, y)); w = list(map(float, w))
    val, wt = [], []
    for yi, wi in zip(y, w):
        val.append(yi); wt.append(wi)
        # vi pham khi block truoc THAP hon block sau (ta muon khong tang)
        while len(val) > 1 and val[-2] < val[-1]:
            v2, w2 = val.pop(), wt.pop()
            v1, w1 = val.pop(), wt.pop()
            val.append((v1 * w1 + v2 * w2) / (w1 + w2)); wt.append(w1 + w2)
    out = []
    for v, ww in zip(val, wt):
        out.extend([v] * int(round(ww)))
    return out


def h50_isotonic(heights, hits):
    """h50 tu hoi quy don dieu tren tung hop, khong chia bin.

    Sap theo chieu cao GIAM dan roi ep recall khong tang, tuc gia thiet "cay cang
    nho cang kho phat hien". Sau do doc chieu cao tai cho duong cong cat 0,5.
    """
    h = np.asarray(heights, dtype=float); y = np.asarray(hits, dtype=float)
    o = np.argsort(-h)                     # cao -> thap
    h, y = h[o], y[o]
    fit = np.asarray(pava(y, np.ones_like(y)))
    below = np.flatnonzero(fit < 0.5)
    if below.size == 0:
        return float("nan")                # khong bao gio tut duoi 0,5
    i = below[0]
    if i == 0:
        return float(h[0])
    y1, y2 = fit[i - 1], fit[i]
    if y1 == y2:
        return float(h[i])
    t = (y1 - 0.5) / (y1 - y2)
    return float(h[i - 1] + t * (h[i] - h[i - 1]))


def collect(model_path, items, imgsz, conf, iou, max_det, device, plant_class):
    from ultralytics import YOLO
    model = YOLO(model_path)
    H, HIT, IMG = [], [], []
    for k, (ip, w, h, gt, ig) in enumerate(items):
        r = model.predict(ip, conf=conf, iou=0.6, imgsz=imgsz, device=device,
                          max_det=max_det, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            cl = r.boxes.cls.cpu().numpy()
            pred = r.boxes.xyxy.cpu().numpy()[cl == plant_class]
        else:
            pred = np.zeros((0, 4))
        M = np.array([[box_iou(g[:4], p) for p in pred] for g in gt]) if len(gt) and len(pred) \
            else np.zeros((len(gt), len(pred)))
        hit = set()
        if M.size:
            pairs = sorted(((M[a, b], a, b) for a in range(len(gt))
                            for b in range(pred.shape[0]) if M[a, b] >= iou), reverse=True)
            ug, up = set(), set()
            for _, a, b in pairs:
                if a in ug or b in up:
                    continue
                ug.add(a); up.add(b); hit.add(a)
        for i, gb in enumerate(gt):
            H.append(gb[3] - gb[1]); HIT.append(int(i in hit)); IMG.append(k)
    del model
    return H, HIT, IMG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", help="CSV *_phys.csv da bin san -> chi chay do nhay min_n")
    ap.add_argument("--minn-grid", nargs="+", type=int,
                    default=[0, 10, 25, 50, 100, 200, 500, 1000])
    ap.add_argument("--labels-dir")
    ap.add_argument("--images-dir")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="stock")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--imgsz-list", nargs="+", type=int, default=[1280, 1920, 2560, 3840])
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=3000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.from_csv:
        rows = sensitivity(a.from_csv, a.minn_grid)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"[OK] do nhay min_n -> {a.out}")
        cur = None
        for r in rows[1:]:
            if r[0] != cur:
                cur = r[0]; print(f"  imgsz {cur}:")
            print(f"     min_n {r[1]:>5}  bin dung {r[2]}  h50 = {r[3]}")
        return

    import cv2
    idx = build_image_index(a.images_dir)
    items = []
    for lp in sorted(glob.glob(os.path.join(a.labels_dir, "*.txt"))):
        ip = idx.get(os.path.splitext(os.path.basename(lp))[0])
        if ip is None:
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        gt, ig = read_gt(lp, w, h, a.plant_class, a.ignore_class)
        items.append((ip, w, h, gt, ig))
    print(f"[i] {len(items)} anh")

    cps = [os.path.join(a.runs, f"{a.tag}_s{s}", "weights", "best.pt") for s in a.seeds]
    cps = [c for c in cps if os.path.exists(c)]
    rng = np.random.default_rng(0)
    rows = [["imgsz", "n_seeds", "n_box", "h50_binned_minn100", "h50_isotonic",
             "iso_boot_lo", "iso_boot_hi"]]
    for z in a.imgsz_list:
        H, HIT, IMG = [], [], []
        for cp in cps:
            h_, y_, i_ = collect(cp, items, z, a.conf, a.iou, a.max_det,
                                 a.device, a.plant_class)
            H += h_; HIT += y_; IMG += i_
        H = np.asarray(H); HIT = np.asarray(HIT); IMG = np.asarray(IMG)
        # ban da bin, de doi chieu voi con so bai dang dung
        mids, rec, ns = [], [], []
        for i in range(len(PHYS_EDGES) - 1):
            lo, hi = PHYS_EDGES[i], PHYS_EDGES[i + 1]
            m = (H >= lo) & (H < hi)
            rec.append(HIT[m].mean() if m.any() else float("nan"))
            ns.append(int(m.sum() // max(1, len(cps))))
        hb = h50(PHYS_EDGES, rec, ns, 100)
        hi_ = h50_isotonic(H, HIT)

        uniq = np.unique(IMG)
        where = {k: np.flatnonzero(IMG == k) for k in uniq}
        bs = []
        for _ in range(a.boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([where[k] for k in pick])
            v = h50_isotonic(H[sel], HIT[sel])
            if np.isfinite(v):
                bs.append(v)
        lo_, hi2 = (np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"),) * 2)
        rows.append([z, len(cps), len(H) // max(1, len(cps)),
                     round(float(hb), 2), round(float(hi_), 2),
                     round(float(lo_), 2), round(float(hi2), 2)])
        print(f"  imgsz {z:>4}: binned(min_n=100) {hb:6.2f} | isotonic {hi_:6.2f} "
              f"[{lo_:.2f}, {hi2:.2f}]")

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[OK] -> {a.out}")


if __name__ == "__main__":
    main()
