#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hieu so h50 giua HAI nhanh, bootstrap ghep cap tren cung mau cay.

  python exp_h50_contrast.py --labels-dir data/test/labels --images-dir data/test/images \\
      --runs runs --tag-a nwd --tag-b stock --seeds 0 1 2 3 4 --imgsz 1280 \\
      --device 0 --out results/rebuttal/M8_h50_contrast.csv

Vi sao can rieng script nay thay vi doc hai file h50_ci.csv. Hai khoang tin cay roi
nhau thi ket luan duoc la khac nhau, nhung hai khoang CHONG NHAU thi khong ket luan
duoc gi -- do la loi doc khoang pho bien nhat. Phep dung la bootstrap chinh HIEU SO
tren cung mot lan lay mau lai: cay nao vao mau thi vao cho ca hai nhanh, nen phuong
sai chung do mau bi khu di va khoang hep hon han khoang cua tung nhanh.

Don vi lay mau lai la CAY (nhu exp_h50_ci.py), khong phai hop: mot cay vat ly cho
nhieu hop nen coi chung doc lap se thu hep khoang mot cach gia tao.

Bao ca hai nguong tin cay:

  conf=0.25   diem van hanh cua bai. h50 o day tron lan KHA NANG PHAT HIEN voi HIEU
              CHUAN DIEM SO -- mot nhanh cham diem hao phong hon se co h50 thap hon
              ma khong thuc su nhin duoc xa hon.
  conf=0.001  tran. Gan nhu moi du doan deu qua nguong, nen h50 o day do kha nang
              phat hien gan nhu thuan tuy.

Neu hieu so con o CA HAI nguong thi no la that. Neu no bien mat o tran thi cai ta
thay o 0,25 chi la hieu chuan.
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
from exp_resolution_sweep import PHYS_EDGES
from exp_h50_ci import clip_of, iou_matrix, load_ledgers, h50_from


def hits_for(tag, seeds, runs, items, imgsz, conf, iou, max_det, device, plant_class):
    """Vector hit (0/1) song song voi danh sach hop, gop qua cac seed."""
    from ultralytics import YOLO
    cps = [os.path.join(runs, f"{tag}_s{s}", "weights", "best.pt") for s in seeds]
    cps = [c for c in cps if os.path.exists(c)]
    if not cps:
        raise SystemExit(f"khong co checkpoint cho {tag}")
    HIT = []
    for cp in cps:
        model = YOLO(cp)
        for ip, w_img, h_img, gt, ig, pid, clip in items:
            r = model.predict(ip, conf=conf, iou=0.6, imgsz=imgsz, device=device,
                              max_det=max_det, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                cl = r.boxes.cls.cpu().numpy()
                pred = r.boxes.xyxy.cpu().numpy()[cl == plant_class]
            else:
                pred = np.zeros((0, 4))
            M = iou_matrix(gt, pred)
            hit = set()
            if M.size:
                pairs = sorted(((M[g, j], g, j) for g in range(len(gt))
                                for j in range(pred.shape[0]) if M[g, j] >= iou), reverse=True)
                ug, up = set(), set()
                for _, g, j in pairs:
                    if g in ug or j in up:
                        continue
                    ug.add(g); up.add(j); hit.add(g)
            HIT.extend(int(i in hit) for i in range(len(gt)))
        del model
    return np.asarray(HIT), len(cps)


def build_items(labels_dir, images_dir, ledger_dir, link_iou, plant_class):
    import cv2
    idx = build_image_index(images_dir)
    led = load_ledgers(ledger_dir)
    by_clip = defaultdict(list)
    for lp in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        by_clip[clip_of(lp)].append(lp)
    items = []
    for clip, lps in by_clip.items():
        for slot, lp in enumerate(sorted(lps)):
            ip = idx.get(os.path.splitext(os.path.basename(lp))[0])
            if ip is None:
                continue
            im = cv2.imread(ip)
            if im is None:
                continue
            h_img, w_img = im.shape[:2]
            gt, ig = read_gt(lp, w_img, h_img, plant_class)
            cand = led.get(clip, {}).get(slot, [])
            pid = [None] * len(gt)
            if cand and len(gt):
                M = iou_matrix(gt, np.array([c[1] for c in cand]))
                pairs = sorted(((M[g, j], g, j) for g in range(len(gt))
                                for j in range(len(cand)) if M[g, j] >= link_iou), reverse=True)
                ug, up = set(), set()
                for _, g, j in pairs:
                    if g in ug or j in up:
                        continue
                    ug.add(g); up.add(j); pid[g] = f"{clip}#{cand[j][0]}"
            for i in range(len(gt)):
                if pid[i] is None:
                    pid[i] = f"{clip}#solo{slot}_{i}"
            items.append((ip, w_img, h_img, gt, ig, pid, clip))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--ledger-dir", default="results/counting")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag-a", required=True)
    ap.add_argument("--tag-b", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--imgsz", type=int, default=1280)
    # De hai ve khac imgsz: dat --tag-a = --tag-b va cho hai --imgsz-* khac nhau.
    # Khi do phep thu tra loi "them diem anh dau vao co doi san khong", tren cung
    # mot nhanh va cung mot mau cay -- dung cau hoi cua C1.
    ap.add_argument("--imgsz-a", type=int, default=None)
    ap.add_argument("--imgsz-b", type=int, default=None)
    ap.add_argument("--conf-list", nargs="+", type=float, default=[0.25, 0.001])
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--link-iou", type=float, default=0.3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--out", default="results/rebuttal/M8_h50_contrast.csv")
    a = ap.parse_args()

    items = build_items(a.labels_dir, a.images_dir, a.ledger_dir, a.link_iou, a.plant_class)
    H = np.array([gb[3] - gb[1] for _, _, _, gt, _, _, _ in items for gb in gt])
    PID = np.array([p for _, _, _, _, _, pid, _ in items for p in pid])
    print(f"[i] {len(items)} anh | {len(H)} hop | {len(np.unique(PID))} cay")

    za = a.imgsz_a if a.imgsz_a is not None else a.imgsz
    zb = a.imgsz_b if a.imgsz_b is not None else a.imgsz
    side_a, side_b = (a.tag_a, za), (a.tag_b, zb)
    if side_a == side_b:
        raise SystemExit("hai ve trung nhau: doi --tag-* hoac --imgsz-*")

    rng = np.random.default_rng(0)
    rows = [["conf", "tag_a", "imgsz_a", "tag_b", "imgsz_b", "n_seeds", "n_plant",
             "h50_a", "h50_b", "delta", "delta_lo", "delta_hi", "p_boot", "ket_luan"]]
    for conf in a.conf_list:
        hits = {}
        for tag, z in (side_a, side_b):
            if (tag, z) in hits:
                continue
            h, ns = hits_for(tag, a.seeds, a.runs, items, z, conf, a.iou,
                             a.max_det, a.device, a.plant_class)
            hits[(tag, z)] = h.reshape(ns, -1)     # (seed, hop)
        HA, HB = hits[side_a], hits[side_b]
        nseed = HA.shape[0]
        Hs = np.tile(H, nseed)
        h50a = h50_from(Hs, HA.ravel(), PHYS_EDGES)
        h50b = h50_from(Hs, HB.ravel(), PHYS_EDGES)

        uniq = np.unique(PID)
        where = {k: np.flatnonzero(PID == k) for k in uniq}
        ds = []
        for _ in range(a.boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([where[k] for k in pick])
            # Cung mot mau cay cho ca hai nhanh -> phuong sai chung do mau bi khu.
            sel_all = np.concatenate([sel + s * len(H) for s in range(nseed)])
            hs = Hs[sel_all]
            va = h50_from(hs, HA.ravel()[sel_all], PHYS_EDGES)
            vb = h50_from(hs, HB.ravel()[sel_all], PHYS_EDGES)
            if np.isfinite(va) and np.isfinite(vb):
                ds.append(va - vb)
        ds = np.asarray(ds)
        lo, hi = np.percentile(ds, [2.5, 97.5])
        # p hai phia tu phan phoi bootstrap cua hieu so
        p = 2 * min((ds >= 0).mean(), (ds <= 0).mean())
        p = min(1.0, max(p, 1.0 / max(len(ds), 1)))
        verdict = "KHAC" if (lo > 0 or hi < 0) else "khong phan biet duoc"
        la, lb = f"{a.tag_a}@{za}", f"{a.tag_b}@{zb}"
        rows.append([conf, a.tag_a, za, a.tag_b, zb, nseed, len(uniq),
                     round(h50a, 2), round(h50b, 2), round(h50a - h50b, 2),
                     round(lo, 2), round(hi, 2), round(float(p), 4), verdict])
        print(f"  conf {conf:<6}: h50({la})={h50a:6.2f}  h50({lb})={h50b:6.2f}"
              f"  delta={h50a - h50b:+6.2f} [{lo:+.2f}, {hi:+.2f}]  p={p:.4f}  {verdict}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[OK] -> {a.out}")


if __name__ == "__main__":
    main()
