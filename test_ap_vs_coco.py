#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Doi chieu ap_tier voi pycocotools tren cac truong hop tong hop.

  python test_ap_vs_coco.py

Bai bao cao AP theo tang va noi la theo quy uoc COCO. Cau do chi dung neu code that
su khop pycocotools, va khong cach nao biet duoc ngoai viec chay ca hai tren cung mot
du lieu. Test nay dung cac canh tong hop nho, dat nhan va du doan bang tay, roi so
AP cua `ap_tier` voi `COCOeval` o cung nguong IoU va cung dai kich thuoc.

Ba diem tung sai va nay duoc chot lai o day:

  1. Detection khong khop nhung NAM NGOAI dai kich thuoc phai bi bo qua, khong tinh
     la false positive. Thieu luat nay thi AP tang xa dem ca box qua kho.
  2. Detection duoc xep theo CONFIDENCE, khong phai theo IoU.
  3. Tich phan la noi suy 101 diem cua COCO, khong phai all-point cua VOC2010+.

Chay khong can GPU va khong can du lieu that.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ap import ap_tier                                          # noqa: E402

# Bai dung chieu cao POT de phan tang; COCO dung dien tich. De so duoc, ta lam moi
# box VUONG, khi do dien tich = chieu cao^2 va hai cach phan tang trung nhau.
TIER_EDGES = {"far": (0.0, 16.0), "mid": (16.0, 32.0), "near": (32.0, 1e5)}


def square(cx, cy, h):
    return [cx - h / 2, cy - h / 2, cx + h / 2, cy + h / 2]


def build(case):
    """case -> (per_image cho ap_tier, coco_gt dict, coco_dt list)."""
    per_image, images, anns, dets = [], [], [], []
    aid = 1
    for i, (gt_boxes, preds) in enumerate(case):
        images.append({"id": i, "width": 1920, "height": 1080})
        gts = []
        for (b, h) in gt_boxes:
            t = "far" if h < 16 else ("mid" if h < 32 else "near")
            gts.append((b, t))
            anns.append({"id": aid, "image_id": i, "category_id": 1, "iscrowd": 0,
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "area": (b[2] - b[0]) * (b[3] - b[1])})
            aid += 1
        for (c, b) in preds:
            dets.append({"image_id": i, "category_id": 1, "score": c,
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
        per_image.append((gts, [], preds))
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": 1, "name": "plant"}]}
    return per_image, gt, dets


def coco_ap_ref(gt, dets, iou_thr, tier):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    lo, hi = TIER_EDGES[tier]
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "gt.json")
        json.dump(gt, open(gp, "w"))
        c = COCO(gp)
        cd = c.loadRes(dets) if dets else None
        if cd is None:
            return float("nan")
        e = COCOeval(c, cd, "bbox")
        e.params.iouThrs = np.array([iou_thr])
        e.params.areaRng = [[lo * lo, hi * hi]]
        e.params.areaRngLbl = [tier]
        e.params.maxDets = [1000]
        e.evaluate(); e.accumulate()
        p = e.eval["precision"][0, :, 0, 0, 0]
        p = p[p > -1]
        return float(p.mean()) if p.size else 0.0


CASES = {
    # mot cay xa, bat dung -> AP = 1
    "far hoan hao": [([(square(100, 100, 12), 12)], [(0.9, square(100, 100, 12))])],
    # mot cay xa, bat dung, cong mot box GAN doan sai.
    # COCO bo qua box gan (ngoai dai kich thuoc) -> AP van = 1.
    "far + FP co gan": [([(square(100, 100, 12), 12)],
                          [(0.9, square(100, 100, 12)), (0.8, square(600, 600, 60))])],
    # mot cay xa, bat dung, cong mot box XA doan sai -> AP giam that
    "far + FP co xa": [([(square(100, 100, 12), 12)],
                         [(0.9, square(100, 100, 12)), (0.8, square(600, 600, 12))])],
    # thu tu confidence quan trong: box sai co diem cao hon box dung
    "thu tu confidence": [([(square(100, 100, 12), 12)],
                            [(0.95, square(600, 600, 12)), (0.5, square(100, 100, 12))])],
    # hai anh, mot bat duoc mot khong -> recall 0.5
    "hai anh, recall 0.5": [([(square(100, 100, 12), 12)], [(0.9, square(100, 100, 12))]),
                             ([(square(200, 200, 12), 12)], [])],
    # cay tang khac trung hoa du doan
    "GT tang khac": [([(square(100, 100, 12), 12), (square(600, 600, 60), 60)],
                       [(0.9, square(100, 100, 12)), (0.8, square(600, 600, 60))])],
}


def main():
    iou = 0.3
    bad = 0
    print(f"  {'truong hop':24s} {'tang':5s} {'ap_tier':>9s} {'pycocotools':>12s}")
    for name, case in CASES.items():
        per_image, gt, dets = build(case)
        for tier in ("far", "near"):
            if not any(t == tier for gts, _, _ in per_image for _, t in gts):
                continue
            scale = 1.0                      # box da o toa do goc, khong resize
            mine, n = ap_tier(per_image, tier, iou, fp_scale=scale)
            ref = coco_ap_ref(gt, dets, iou, tier)
            ok = abs(mine - ref) < 0.02      # 101 diem -> sai so roi rac ~0.01
            bad += (not ok)
            print(f"  {name:24s} {tier:5s} {mine:>9.4f} {ref:>12.4f}  {'OK' if ok else 'LECH'}")
    print(f"\n  {'tat ca khop pycocotools' if not bad else str(bad) + ' truong hop LECH'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
