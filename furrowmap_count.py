#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count-when-near baseline, scored against the verified ledger.

  B0a  single frame: the largest plant count in any one frame (undercounts)
  B0b  gross: total detections over all frames (no dedup, overcounts)
  B1   ByteTrack IDs: confirmed track identities (>= min-frames, conf >= confirm) -- the baseline

Reports count error, per-plant precision/recall/F1, double-counts and merges. Predicted tracks are
matched to ledger plants at labelled frames by stem-base distance.

  python furrowmap_count.py --weights runs/nearfar_s0/weights/best.pt \
      --frames-dir data/images/IMG_3916_test --ledger results/counting/ledger_IMG_3916.json \
      --imgsz 1280 --device 0
"""
import argparse
import glob
import json
import os

import numpy as np

from tbxrd_mint import track_and_detect


def greedy_match(overlap):
    """Greedy one-to-one matching by number of co-occurring frames."""
    pairs = sorted(((overlap[i][j], i, j) for i in range(len(overlap)) for j in range(len(overlap[0]) if overlap else 0)
                    if overlap[i][j] > 0), reverse=True)
    up, ug, m = set(), set(), {}
    for ov, i, j in pairs:
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); m[i] = j
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--ledger", required=True, help="ledger JSON tu furrowmap_ledger.py")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25, help="tracker confidence")
    ap.add_argument("--min-frames", type=int, default=3, help="confirm: track phai >= so khung")
    ap.add_argument("--conf-confirm", type=float, default=0.4, help="confirm: max conf >= nguong")
    ap.add_argument("--tau", type=float, default=55.0, help="stem-base distance (px) below which two detections are the same plant")
    ap.add_argument("--min-overlap", type=int, default=1, help="minimum number of shared labelled frames for a match")
    ap.add_argument("--min-pot-reach", type=float, default=32.0,
                    help="count only tracks that reach this POT, i.e. plants that pass the resolvable near zone. "
                         "Nen KHOP min-pot-reach cua ledger de GT/pred cung scope.")
    a = ap.parse_args()

    led = json.load(open(a.ledger, encoding="utf-8"))
    gt_plants = led["plants"]; gt_count = led["n_plants"]
    labeled_frames = sorted({s[0] for p in gt_plants for s in p["stems"]})
    # GT stems per (plant, frame)
    gt_stem = [{s[0]: (s[1], s[2]) for s in p["stems"]} for p in gt_plants]

    from ultralytics import YOLO
    model = YOLO(a.weights)
    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")) +
                    glob.glob(os.path.join(a.frames_dir, "*.png")) +
                    glob.glob(os.path.join(a.frames_dir, "*.jpeg")))
    a.imgsz = a.imgsz
    tracks, frame_dets, HW = track_and_detect(model, frames, a)
    H, W = HW
    scale = a.imgsz / max(W, H)                                     # px goc -> px input (cho POT chieu cao)

    # ---- baselines dem ----
    per_frame_counts = [len(v) for v in frame_dets.values()]
    b0a = max(per_frame_counts) if per_frame_counts else 0          # single-frame max
    b0b = sum(per_frame_counts)                                     # gross
    conf_tracks = {tid: seq for tid, seq in tracks.items()
                   if len(seq) >= a.min_frames and max(s[5] for s in seq) >= a.conf_confirm
                   and max(s[4] for s in seq) * scale >= a.min_pot_reach}   # co luc qua vung GAN
    b1 = len(conf_tracks)

    # ---- predicted-track stems at labelled frames ----
    lf = set(labeled_frames)
    pred_stem = []   # list theo track: {frame: (cx,y2)}
    pids = []
    for tid, seq in conf_tracks.items():
        d = {idx: (cx, y2) for (idx, cx, y2, w, h, c, pot) in seq if idx in lf}
        if d:
            pred_stem.append(d); pids.append(tid)

    # ---- overlap matrix (predicted x GT) at labelled frames ----
    P, G = len(pred_stem), len(gt_plants)
    overlap = [[0] * G for _ in range(P)]
    for i in range(P):
        for j in range(G):
            ov = 0
            for f, (px, py) in pred_stem[i].items():
                if f in gt_stem[j]:
                    gx, gy = gt_stem[j][f]
                    if ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 <= a.tau:
                        ov += 1
            overlap[i][j] = ov if ov >= a.min_overlap else 0

    m = greedy_match(overlap)
    TP = len(m); FP = P - TP; FN = G - len(set(m.values()))
    prec = TP / (TP + FP) if TP + FP else float("nan")
    rec = TP / (TP + FN) if TP + FN else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not (np.isnan(prec) or np.isnan(rec)) else float("nan")
    # double count: a GT plant claimed by two or more predicted tracks
    gt_cands = [sum(1 for i in range(P) if overlap[i][j] > 0) for j in range(G)]
    double = sum(1 for c in gt_cands if c >= 2)
    # merge: one predicted track spanning two or more GT plants
    pred_cands = [sum(1 for j in range(G) if overlap[i][j] > 0) for i in range(P)]
    merge = sum(1 for c in pred_cands if c >= 2)

    print(f"\n=== {led['clip']} | GT unique = {gt_count} plants ({led['n_labeled']} labelled frames) ===")
    print(f"  B0a single-frame max : {b0a:>5}   (sot: chi 1 khung)")
    print(f"  B0b gross (no dedup) : {b0b:>5}   (over-count khung)")
    print(f"  B1  ByteTrack-ID     : {b1:>5}   <- baseline chinh")
    err = b1 - gt_count
    print(f"\n  COUNT: pred(B1)={b1}  gt={gt_count}  err={err:+d}  |err|%={abs(err)/max(gt_count,1)*100:.1f}%")
    print(f"  per-plant: P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}   (TP={TP} FP={FP} FN={FN})")
    print(f"  Double-count(GT bi tach)={double}   Merge(pred om >1 GT)={merge}   tau={a.tau:.0f}px")
    print("\n  # a BEV re-association baseline should reduce both error and double counts.")


if __name__ == "__main__":
    main()
