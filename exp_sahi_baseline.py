#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sliced (SAHI-style) inference: does tiling recover the far tier?

The paper argues in-frame processing cannot create information the sensor never captured. This
tests the most obvious counterexample. Each 1920x1080 frame is cut into tiles (e.g. 640 px) that
are run at imgsz 1280, so a far plant is presented at roughly twice its usual POT. Tile boxes are
offset back to frame coordinates, merged by NMS, and optionally unioned with a full-frame pass.
Scoring is the same ignore-aware per-tier AP as eval_ap.py, so the numbers compare directly.

  python exp_sahi_baseline.py --labels-dir data/test/labels --images-dir data/test/images \
      --runs runs --tag nearfar --seeds 0 1 2 --tile 640 --overlap 0.2 --imgsz 1280 \
      --iou 0.3 --device 0 --out results/baselines/sahi_ap.csv
  python exp_sahi_baseline.py --selftest     # logic check, no GPU needed
"""
import argparse
import csv
import glob
import os

import numpy as np

from eval_testset import read_gt, build_image_index, stratum, in_any, sstd
from eval_ap import box_iou, ap_tier, predictions_for

STRATA = ["near", "mid", "far", "all"]
IMGSZ_REF = 1280   # POT/tier tham chieu (giong baseline) -> tier khong doi theo tiling


def nms(boxes, iou_thr):
    """Greedy NMS over (conf, box) pairs, highest confidence first."""
    keep = []
    for c, b in sorted(boxes, key=lambda z: -z[0]):
        if all(box_iou(b, kb) < iou_thr for _, kb in keep):
            keep.append((c, b))
    return keep


def tile_grid(W, H, ts, ov):
    """Top-left corners of ts x ts tiles with overlap ov, covering the whole frame."""
    step = max(1, int(round(ts * (1 - ov))))
    xs = list(range(0, max(1, W - ts + 1), step)) or [0]
    ys = list(range(0, max(1, H - ts + 1), step)) or [0]
    if xs[-1] + ts < W:
        xs.append(max(0, W - ts))
    if ys[-1] + ts < H:
        ys.append(max(0, H - ts))
    xs = sorted(set(min(x, max(0, W - ts)) for x in xs))
    ys = sorted(set(min(y, max(0, H - ts)) for y in ys))
    return [(x, y) for y in ys for x in xs]


def fp_scale_of(items):
    """He so doi chieu cao pixel goc sang POT, dung cho bo loc kich thuoc FP.

    PHAI trung voi he so dung de phan tang ground truth o `sahi_per_image`, tuc
    IMGSZ_REF / max(W, H) cua CHINH bo anh dang cham. Truoc day cho nay cung hoa
    1920 nen tren bo 8K no lech dung 4 lan: ground truth tang xa la hop cao duoi
    96 px goc (16 / (1280/7680)), con bo loc lai chi coi mot false positive la
    "tang xa" khi no thap hon 24 px (16 / (1280/1920)). Moi false positive cao
    24--96 px bi bo qua, nen AP tang xa tren 8K bi thoi phong, va doi chung E5
    khong con cong bang vi nhanh ha mau 1920x1080 lai duoc cham dung he so.

    Bo anh phai dong nhat kich thuoc; neu khong thi mot he so vo huong la sai va
    ta dung han thay vi cham nham.
    """
    sizes = {(w, h) for _, w, h, _, _ in items}
    if len(sizes) != 1:
        raise SystemExit(f"anh khong dong nhat kich thuoc ({len(sizes)} co), "
                         "khong dung duoc mot he so fp_scale vo huong")
    w, h = sizes.pop()
    return IMGSZ_REF / max(w, h)


def tile_predict(model, ip, W, H, args):
    """Tiled inference -> NMS-merged predictions in frame coordinates."""
    import cv2
    im = cv2.imread(ip)
    ts = min(args.tile, W, H)
    boxes = []
    for (x0, y0) in tile_grid(W, H, ts, args.overlap):
        tile = im[y0:y0 + ts, x0:x0 + ts]
        r = model.predict(tile, conf=args.conf, iou=0.6, imgsz=args.imgsz,
                          device=args.device, max_det=args.max_det, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            cl = r.boxes.cls.cpu().numpy(); xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            for j in range(len(cl)):
                if int(cl[j]) == args.plant_class:
                    b = xy[j]
                    boxes.append((float(cf[j]), [float(b[0]) + x0, float(b[1]) + y0,
                                                 float(b[2]) + x0, float(b[3]) + y0]))
    if args.add_full:
        r = model.predict(ip, conf=args.conf, iou=0.6, imgsz=args.imgsz,
                          device=args.device, max_det=args.max_det, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            cl = r.boxes.cls.cpu().numpy(); xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
            for j in range(len(cl)):
                if int(cl[j]) == args.plant_class:
                    b = xy[j]
                    boxes.append((float(cf[j]), [float(b[0]), float(b[1]), float(b[2]), float(b[3])]))
    merged = nms(boxes, args.nms_iou)
    # Cat ve tran o muc KHUNG. `max_det` o tren ap cho TUNG o, nen khong co buoc nay
    # thi nhanh cat o duoc ngan sach gap so-o lan nhanh toan khung (~28 o tren bo 8K),
    # va cau "hai nhanh dung cung mot tran phat hien" trong bai la sai. Giu box theo
    # do tin cay giam dan, giong cach Ultralytics ap tran cho mot lan suy luan.
    cap = args.frame_max_det if args.frame_max_det is not None else args.max_det
    if cap and len(merged) > cap:
        merged = sorted(merged, key=lambda cb: -cb[0])[:cap]
    return merged


def sahi_per_image(model_path, items, args):
    """per_image [(gts_with_tier, ignores, preds)] using tiled predictions."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    out = []
    for ip, w, h, plants, ignores in items:
        scale = IMGSZ_REF / max(w, h)                       # tier o imgsz tham chieu (giong baseline)
        gts = [(b[:4], stratum(b, "pot", scale)) for b in plants]
        preds = tile_predict(model, ip, w, h, args)
        out.append((gts, [b[:4] for b in ignores], preds))
    return out


def run_selftest():
    ok = []

    def chk(name, cond):
        ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    # 1) the tile grid covers the frame and stays in bounds
    g = tile_grid(1920, 1080, 640, 0.2)
    chk("tile_grid phu canh phai/duoi", any(x + 640 >= 1920 for x, y in g) and any(y + 640 >= 1080 for x, y in g))
    chk("tile_grid goc trong bien", all(0 <= x <= 1920 - 640 and 0 <= y <= 1080 - 640 for x, y in g))
    # 2) NMS merges duplicates
    b = [(0.9, [0, 0, 10, 10]), (0.8, [1, 1, 11, 11]), (0.7, [100, 100, 110, 110])]
    k = nms(b, 0.5)
    chk("nms bo box trung (giu 2/3)", len(k) == 2 and k[0][0] == 0.9)
    chk("nms giu box conf cao nhat", k[0][1] == [0, 0, 10, 10])
    # 3) ap_tier (reused from eval_ap): one hit on one near GT -> AP=1
    per = [([([10, 10, 40, 60], "near")], [], [(0.9, [10, 10, 40, 60])])]
    ap, ng = ap_tier(per, "near", 0.3)
    chk("ap_tier perfect near -> AP~1", abs(ap - 1.0) < 1e-6 and ng == 1)
    # 4) tile-local box + tile origin = frame coordinates
    x0, y0 = 640, 320; bl = [5, 6, 25, 46]
    bg = [bl[0] + x0, bl[1] + y0, bl[2] + x0, bl[3] + y0]
    chk("offset box dung", bg == [645, 326, 665, 366])
    # 5) he so loc FP phai TRUNG he so phan tang ground truth, tren MOI co anh.
    #    Day la loi da xay ra that: fp_scale cung hoa 1920 trong khi ground truth
    #    phan tang theo max(W,H), nen tren bo 8K nguong lech dung 4 lan va AP tang
    #    xa bi thoi phong. Kiem ca hai co de mot bo dung khong che lap bo kia.
    for (W, H) in ((1920, 1080), (7680, 4320)):
        fake = [("x.jpg", W, H, [], [])]
        chk(f"fp_scale khop scale phan tang o {W}x{H}",
            abs(fp_scale_of(fake) - IMGSZ_REF / max(W, H)) < 1e-12)
    # 6) tran o muc khung phai duoc ap sau khi gop o
    import inspect
    src = inspect.getsource(tile_predict)
    chk("tile_predict co cat ve tran muc khung", "frame_max_det" in src)

    print(f"\n[SELFTEST] {sum(ok)}/{len(ok)} PASS")
    if not all(ok):
        raise SystemExit("[SELFTEST] FAILED")
    print("[SELFTEST] logic tiling/NMS/scoring OK.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir")
    ap.add_argument("--images-dir")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="nearfar")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tile", type=int, default=640, help="tile side in native px; smaller means more upscaling")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1280, help="imgsz used per tile; tile < imgsz means upscaling")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.6)
    ap.add_argument("--add-full", action="store_true", default=True, help="union tiled with full-frame predictions (best case)")
    ap.add_argument("--no-add-full", dest="add_full", action="store_false")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=1000, help="tran cho MOI o")
    ap.add_argument("--frame-max-det", type=int, default=None,
                    help="tran sau khi gop o; mac dinh bang --max-det de khop nhanh toan khung. "
                         "Dat 0 de bo cap (hanh vi cu, KHONG khop tran).")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--weight", default="best.pt", choices=["best.pt", "last.pt"])
    ap.add_argument("--out", default="sahi_ap.csv")
    ap.add_argument("--fp-size-filter", action="store_true",
                    help="chi tinh FP cua mot tang khi prediction thuoc tang do (kieu COCO)")
    ap.add_argument("--per-seed-out", default=None,
                    help="ghi them AP tho theo tung seed, de chay Welch/phep kiem tuong tac")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(); return
    if not args.labels_dir or not args.images_dir:
        raise SystemExit("--labels-dir and --images-dir are required unless --selftest")

    import cv2
    img_idx = build_image_index(args.images_dir)
    items = []
    for lp in sorted(glob.glob(os.path.join(args.labels_dir, "**", "*.txt"), recursive=True)):
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = img_idx.get(stem)
        if ip is None:
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        plants, ignores = read_gt(lp, w, h, args.plant_class, args.ignore_class)
        items.append((ip, w, h, plants, ignores))
    if not items:
        raise SystemExit("No label could be paired with an image.")
    print(f"[i] {len(items)} images | tag={args.tag} | tile={args.tile} imgsz={args.imgsz} "
          f"overlap={args.overlap} add_full={args.add_full} | IoU={args.iou}")

    cps = [os.path.join(args.runs, f"{args.tag}_s{s}", "weights", args.weight) for s in args.seeds]
    cps = [c for c in cps if os.path.exists(c)]
    if not cps:
        raise SystemExit(f"No checkpoint for tag={args.tag}")

    # tiled and full-frame predictions from the same model, compared per tier
    sahi = {s: [] for s in STRATA}
    full = {s: [] for s in STRATA}
    ngt = None

    class A:  # shim cho predictions_for (full-image baseline)
        pass
    fa = A()
    for k, v in vars(args).items():
        setattr(fa, k, v)
    fa.imgsz = args.imgsz

    for cp in cps:
        print(f"  [{cp}] tiling ...")
        per_s = sahi_per_image(cp, items, args)
        per_f = predictions_for(cp, items, fa)
        for s in STRATA:
            # fp_scale: chi tinh mot prediction la false positive cua tang dang xet
            # khi chinh no thuoc tang do (kieu COCO). Khong co no, 88,6% FP tinh vao
            # AP tang xa lai la box co kich thuoc tang gan, nen AP tang xa do box lon
            # doan sai chu khong do chat luong phat hien cay xa -- va cat o sinh rat
            # nhieu box lon doan sai, dung o bien o.
            fps = fp_scale_of(items) if args.fp_size_filter else None
            a_s, ng = ap_tier(per_s, s, args.iou, fp_scale=fps)
            a_f, _ = ap_tier(per_f, s, args.iou, fp_scale=fps)
            sahi[s].append(a_s); full[s].append(a_f)
            if ngt is None or s not in (ngt or {}):
                pass
        if ngt is None:
            ngt = {s: ap_tier(per_s, s, args.iou)[1] for s in STRATA}

    rows = [["tier", "n_gt", "full_ap_mean", "full_ap_std", "sahi_ap_mean", "sahi_ap_std", "delta_sahi_minus_full", "n_seeds"]]
    print("\n=== AP per-tier: FULL (baseline) vs SAHI (tiled) ===")
    for s in STRATA:
        fm, fs = float(np.nanmean(full[s])), sstd(full[s])
        sm, ss = float(np.nanmean(sahi[s])), sstd(sahi[s])
        rows.append([s, ngt[s], round(fm, 4), round(fs, 4), round(sm, 4), round(ss, 4), round(sm - fm, 4), len(cps)])
        print(f"  {s:5} (n={ngt[s]:4}): full={fm:.4f}  sahi={sm:.4f}  delta={sm-fm:+.4f}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n-> {args.out}")

    # Ban tong hop chi giu mean/std, khong du de chay Welch hay phep kiem tuong tac
    # giua hai bo du lieu. --per-seed-out ghi lai gia tri tho theo tung seed.
    if args.per_seed_out:
        pr = [["tag", "tier", "seed", "full_ap", "sahi_ap", "delta"]]
        for s in STRATA:
            for k, cp in enumerate(cps):
                sd = os.path.basename(os.path.dirname(os.path.dirname(cp)))
                pr.append([args.tag, s, sd, round(full[s][k], 6), round(sahi[s][k], 6),
                           round(sahi[s][k] - full[s][k], 6)])
        os.makedirs(os.path.dirname(args.per_seed_out) or ".", exist_ok=True)
        with open(args.per_seed_out, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(pr)
        print(f"-> {args.per_seed_out}  (gia tri tung seed)")
    print("# if SAHI far AP stays near zero, tiling and upscaling do not recover the far tier")
    print("#   which supports the sensor-limit reading: more sensor pixels, not more processing.")


if __name__ == "__main__":
    main()
