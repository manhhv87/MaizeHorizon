#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Khoang tin cay cho h50, lay mau lai theo CAY va theo CLIP.

  python exp_h50_ci.py --labels-dir data/test/labels --images-dir data/test/images \\
      --runs runs --tag nearfar --seeds 0 1 2 3 4 --imgsz-list 640 960 1280 1920 \\
      --device 0 --out results/scaling/h50_ci.csv

Vi sao can. `h50` la dai luong trung tam cua bai -- moi khang dinh ve tam lam viec di
qua no -- nhung tu truoc den nay no chi duoc bao cao bang mot con so diem. Khong co
khoang tin cay thi khong tra loi duoc cau hoi hien nhien nhat: chenh lech 1,7 px giua
input 960 va 1920 co phan biet duoc voi khong khong.

Don vi lay mau lai la CAY, khong phai hop va khong phai seed:

  hop   : mot cay vat ly cho nhieu hop (3,6 o tang gan), nen coi chung doc lap se
          thu hep khoang mot cach gia tao.
  seed  : moi seed cham lai DUNG nhung cay ay, nen phuong sai giua seed do lan chay
          chu khong do mau. Da dinh mot lan: phep Welch tren 5 seed bao p=0,017 cho
          mot khac biet ma bootstrap theo cay cho thay nam gon trong khoang.
  cay   : dung don vi. Danh tinh cay lay tu so cai da kiem chung tay, giong
          exp_cluster_ci.py.

Cung bao ca khoang theo CLIP, vi khai quat sang mot lan thu moi thi clip moi la don
vi -- va chi co ba clip, nen khoang do rat rong. Do la su that ve thiet ke, khong
phai khiem khuyet cua phep tinh.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from eval_testset import read_gt, build_image_index
from exp_resolution_sweep import PHYS_EDGES   # dung chung bien bin, khong dinh nghia lai
from eval_ap import box_iou

CLIP_RE = re.compile(r"_f\d+$")


def clip_of(p):
    return CLIP_RE.sub("", os.path.splitext(os.path.basename(p))[0])


def iou_matrix(a, b):
    return np.array([[box_iou(x[:4], y[:4]) for y in b] for x in a]) if len(a) and len(b) \
        else np.zeros((len(a), len(b)))


def load_ledgers(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "ledger_*.json"))):
        if "_near" in f:
            continue
        led = json.load(open(f))
        clip = led["clip"].replace("_test", "")
        frames = sorted({b[0] for p in led["plants"] for b in p["boxes"]})
        slot = {fr: i for i, fr in enumerate(frames)}
        per = defaultdict(list)
        for p in led["plants"]:
            for b in p["boxes"]:
                per[slot[b[0]]].append((p["id"], np.array(b[1:5], dtype=float)))
        out[clip] = per
    return out


def h50_from(heights, hits, edges):
    """h50 noi suy giua diem giua bin, tren mot mau (heights, hits) bat ky."""
    heights = np.asarray(heights)
    hits = np.asarray(hits, dtype=float)
    mids, rec = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (heights >= lo) & (heights < hi)
        if not m.any():
            continue
        mids.append((lo + (hi if np.isfinite(hi) else lo + 32)) / 2)
        rec.append(hits[m].mean())
    prev_m = prev_r = None
    for m, r in zip(mids, rec):
        if r >= 0.5:
            if prev_r is not None and prev_r < 0.5 and m != prev_m:
                t = (0.5 - prev_r) / (r - prev_r)
                return prev_m + t * (m - prev_m)
            return m
        prev_m, prev_r = m, r
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--ledger-dir", default="results/counting")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="nearfar")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--imgsz-list", nargs="+", type=int, default=[640, 960, 1280, 1920])
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--link-iou", type=float, default=0.3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--out", default="results/scaling/h50_ci.csv")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    idx = build_image_index(a.images_dir)
    led = load_ledgers(a.ledger_dir)
    by_clip = defaultdict(list)
    for lp in sorted(glob.glob(os.path.join(a.labels_dir, "*.txt"))):
        by_clip[clip_of(lp)].append(lp)

    items, linked, unlinked = [], 0, 0
    for clip, lps in by_clip.items():
        for slot, lp in enumerate(sorted(lps)):
            ip = idx.get(os.path.splitext(os.path.basename(lp))[0])
            if ip is None:
                continue
            im = cv2.imread(ip)
            if im is None:
                continue
            h_img, w_img = im.shape[:2]
            gt, ig = read_gt(lp, w_img, h_img, a.plant_class)
            cand = led.get(clip, {}).get(slot, [])
            pid = [None] * len(gt)
            if cand and len(gt):
                M = iou_matrix(gt, np.array([c[1] for c in cand]))
                pairs = sorted(((M[g, j], g, j) for g in range(len(gt))
                                for j in range(len(cand)) if M[g, j] >= a.link_iou), reverse=True)
                ug, up = set(), set()
                for _, g, j in pairs:
                    if g in ug or j in up:
                        continue
                    ug.add(g); up.add(j); pid[g] = f"{clip}#{cand[j][0]}"
            for i in range(len(gt)):
                if pid[i] is None:
                    pid[i] = f"{clip}#solo{slot}_{i}"
                    unlinked += 1
                else:
                    linked += 1
            items.append((ip, w_img, h_img, gt, ig, pid, clip))
    print(f"[i] {len(items)} anh | hop noi duoc voi so cai: {linked}, don le: {unlinked}")

    cps = [os.path.join(a.runs, f"{a.tag}_s{s}", "weights", "best.pt") for s in a.seeds]
    cps = [c for c in cps if os.path.exists(c)]
    if not cps:
        raise SystemExit(f"khong co checkpoint cho {a.tag}")
    print(f"[i] {len(cps)} seed | imgsz {a.imgsz_list}")

    rng = np.random.default_rng(0)
    rows = [["imgsz", "n_seeds", "n_box", "n_plant", "n_clip", "h50_mean",
             "plantboot_lo", "plantboot_hi", "clipboot_lo", "clipboot_hi"]]
    for imgsz in a.imgsz_list:
        H, HIT, PID, CLP = [], [], [], []
        for cp in cps:
            model = YOLO(cp)
            for ip, w_img, h_img, gt, ig, pid, clip in items:
                r = model.predict(ip, conf=a.conf, iou=0.6, imgsz=imgsz, device=a.device,
                                  max_det=a.max_det, verbose=False)[0]
                if r.boxes is not None and len(r.boxes):
                    cl = r.boxes.cls.cpu().numpy()
                    pred = r.boxes.xyxy.cpu().numpy()[cl == a.plant_class]
                else:
                    pred = np.zeros((0, 4))
                M = iou_matrix(gt, pred)
                hit = set()
                if M.size:
                    pairs = sorted(((M[g, j], g, j) for g in range(len(gt))
                                    for j in range(pred.shape[0]) if M[g, j] >= a.iou), reverse=True)
                    ug, up = set(), set()
                    for _, g, j in pairs:
                        if g in ug or j in up:
                            continue
                        ug.add(g); up.add(j); hit.add(g)
                for i, gb in enumerate(gt):
                    H.append(gb[3] - gb[1]); HIT.append(int(i in hit))
                    PID.append(pid[i]); CLP.append(clip)
            del model
        H = np.asarray(H); HIT = np.asarray(HIT)
        PID = np.asarray(PID); CLP = np.asarray(CLP)
        point = h50_from(H, HIT, PHYS_EDGES)

        def boot(keys):
            uniq = np.unique(keys)
            where = {k: np.flatnonzero(keys == k) for k in uniq}
            out = []
            for _ in range(a.boot):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                sel = np.concatenate([where[k] for k in pick])
                v = h50_from(H[sel], HIT[sel], PHYS_EDGES)
                if np.isfinite(v):
                    out.append(v)
            return (np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)) if out \
                else (float("nan"), float("nan"))

        plo, phi = boot(PID)
        clo, chi = boot(CLP)
        rows.append([imgsz, len(cps), len(H) // len(cps), len(np.unique(PID)),
                     len(np.unique(CLP)), round(point, 2),
                     round(plo, 2), round(phi, 2), round(clo, 2), round(chi, 2)])
        print(f"  imgsz {imgsz:>4}: h50 = {point:6.2f}  theo cay [{plo:.2f}, {phi:.2f}]"
              f"  theo clip [{clo:.2f}, {chi:.2f}]")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[OK] -> {a.out}")


if __name__ == "__main__":
    main()
