#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track-before-detect feasibility test on the far tier.

Asks whether integrating sub-threshold detector responses along a real trajectory separates plants
from non-plants below the floor (POT < 16 px).

Tracks are anchored on hand-labelled far ground truth, so the sample is genuinely sub-floor, and
decoys are verified positions at least R away from every ground-truth plant, so the control is
clean. For each far box we track backwards with LK, sample the on-track and decoy responses,
integrate over K frames, and measure AUC(on-track vs decoy) per POT bin. A rising AUC with K would
mean temporal integration helps.

  python mve_tbd.py --weights runs/nearfar_s0/weights/best.pt \
      --frames-dir data/images/IMG_3916_test --gt-dir data/test/labels --imgsz 1280 --device 0
"""
import argparse
import glob
import os

import numpy as np

from tbxrd_mint import track_back_lk, track_and_detect

POT_BINS = [(0, 8), (8, 12), (12, 16), (16, 24), (24, 32)]
DECOY_SHIFTS = [2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0]   # be-rong-box, dich ngang; loc bang corn-exclusion
EXCL_R = 36.0   # px: decoy phai cach moi tam cay GT it nhat ngan nay


def rank_auc(pos, neg):
    """Mann-Whitney AUC, with midranks so tied values count as half a win.

    Ties decide the answer in this experiment. If the detector never fires below the floor,
    every on-track and every decoy integral is exactly 0, and the honest statistic is 0.5:
    the two samples are indistinguishable because there is nothing in either of them.
    Ordinal ranks would instead hand the tied positives the lowest ranks (pos is concatenated
    first, and the sort is stable) and report 0.000 -- indistinguishable from decoys genuinely
    beating real plants, which is a different and much stronger claim than the one the data
    support. Use the zero-response diagnostic below to tell an empty integrand from a real one.
    """
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), len(pos)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort"); srt = allv[order]
    ranks = np.empty(len(allv), float)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # midrank shared by the tied block
        i = j + 1
    rpos = ranks[:len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), len(pos)


def bin_of(pot_px):
    for lo, hi in POT_BINS:
        if lo <= pot_px < hi:
            return (lo, hi)
    return None


def read_corn(label_path, W, H, plant_cls=0):
    """Plant boxes in pixels as [x1, y1, x2, y2]."""
    out = []
    if not os.path.exists(label_path):
        return out
    for line in open(label_path, encoding="utf-8"):
        s = line.split()
        if len(s) < 5:
            continue
        if int(float(s[0])) != plant_cls:
            continue
        xc, yc, bw, bh = map(float, s[1:5])
        out.append([(xc - bw / 2) * W, (yc - bh / 2) * H, (xc + bw / 2) * W, (yc + bh / 2) * H])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--gt-dir", required=True, help="hand labels: anchor far GT and exclude plants from the decoys")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--resp-conf", type=float, default=0.02)
    ap.add_argument("--back", type=int, default=40)
    ap.add_argument("--min-px", type=int, default=2)
    ap.add_argument("--green-min", type=float, default=0.10)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--pot-lo", type=float, default=6.0, help="drop far GT too small to track (POT < lo)")
    a = ap.parse_args()
    a.conf = a.resp_conf; a.conf_anchor = 0.5; a.t_near = 0.004; a.min_anchors = 1

    from ultralytics import YOLO
    model = YOLO(a.weights)

    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")) +
                    glob.glob(os.path.join(a.frames_dir, "*.png")) +
                    glob.glob(os.path.join(a.frames_dir, "*.jpeg")))
    if len(frames) < 20:
        raise SystemExit(f"Can frame LIEN TIEP (>=20); thay {len(frames)}")
    stem2idx = {os.path.splitext(os.path.basename(f))[0]: i for i, f in enumerate(frames)}

    # raw per-frame responses from the tracker at a low threshold
    import cv2
    _, frame_dets, HW = track_and_detect(model, frames, a)
    H, W = HW
    scale = a.imgsz / max(W, H)

    def response_at(t, cx, cy, rad):
        best = 0.0
        for (x1, y1, x2, y2, c) in frame_dets.get(t, []):
            ddx = (x1 + x2) / 2 - cx; ddy = (y1 + y2) / 2 - cy
            if ddx * ddx + ddy * ddy <= rad * rad and c > best:
                best = c
        return best

    # collect far GT positives from the hand labels
    gt_files = glob.glob(os.path.join(a.gt_dir, "*.txt"))
    far_seeds = []   # (frame_idx, box, corn_centers_in_that_frame)
    for lp in gt_files:
        stem = os.path.splitext(os.path.basename(lp))[0]
        if stem not in stem2idx:
            continue
        idx = stem2idx[stem]
        corn = read_corn(lp, W, H, a.plant_class)
        corn_c = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in corn]
        for b in corn:
            pot = (b[3] - b[1]) * scale
            if a.pot_lo <= pot < 16:           # far-GT trackable
                far_seeds.append((idx, b, corn_c))
    print(f"[i] {len(frames)} frames | {len(far_seeds)} far-GT seeds (POT {a.pot_lo:.0f}-16px) | scale={scale:.3f}")
    if not far_seeds:
        raise SystemExit("No far GT in [pot_lo, 16). Check --gt-dir and the clip.")

    rows = {b: {k: ([], []) for k in a.ks} for b in POT_BINS}
    n_ok = 0
    for seed_idx, seed_box, corn_c in far_seeds:
        traj = [(seed_idx, seed_box)] + track_back_lk(frames, seed_idx, seed_box, a)
        if len(traj) < 2:
            continue
        # valid decoy shifts: at least EXCL_R away from every GT plant centre
        bw0 = seed_box[2] - seed_box[0]
        cx0 = (seed_box[0] + seed_box[2]) / 2; cy0 = (seed_box[1] + seed_box[3]) / 2
        valid_off = []
        for sh in DECOY_SHIFTS:
            dx = sh * bw0; px = cx0 + dx
            if not (0 <= px <= W):
                continue
            if all(((px - mx) ** 2 + (cy0 - my) ** 2) ** 0.5 >= EXCL_R for (mx, my) in corn_c):
                valid_off.append(dx)
        if not valid_off:
            continue
        n_ok += 1
        on, pots = [], []
        decs = {dx: [] for dx in valid_off}
        for (t, box) in traj:
            bw = box[2] - box[0]; bh = box[3] - box[1]
            cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
            rad = max(bw, bh, 8.0)
            on.append(response_at(t, cx, cy, rad)); pots.append(bh * scale)
            for dx in valid_off:
                decs[dx].append(response_at(t, cx + dx, cy, rad))
        on = np.array(on); pots = np.array(pots)
        decs = {dx: np.array(v) for dx, v in decs.items()}
        for i in range(len(traj)):
            b = bin_of(pots[i])
            if b is None:
                continue
            for k in a.ks:
                if i + 1 < k:
                    continue
                oi = on[i + 1 - k:i + 1].sum()
                di = max(decs[dx][i + 1 - k:i + 1].sum() for dx in valid_off)   # decoy SACH manh nhat
                rows[b][k][0].append(oi); rows[b][k][1].append(di)

    print(f"[i] {n_ok} far GT boxes have a trajectory and a valid decoy.\n")
    print("=== AUC(on-track vs clean decoy) by K, per POT bin (px) ===")
    print(f"{'POT bin':12}" + "".join(f"{'K='+str(k):>10}" for k in a.ks) + f"{'n_pos':>8}")
    go = []
    for b in POT_BINS:
        cells = []; npos = 0
        for k in a.ks:
            au, n = rank_auc(*rows[b][k]); cells.append(au); npos = max(npos, n)
        print(f"{str(b):12}" + "".join((f"{c:>10.3f}" if not np.isnan(c) else f"{'-':>10}") for c in cells) + f"{npos:>8}")
        if b[1] <= 16 and not np.isnan(cells[0]) and not np.isnan(cells[-1]):
            go.append((b, cells[0], cells[-1]))

    # An AUC of 0.5 means "the two samples do not separate", which has two very different
    # causes: a response exists on both sides and does not discriminate, or no response
    # exists at all. Only the second supports the claim that there is nothing to integrate,
    # so report it rather than leaving the reader to infer it from the AUC.
    kmax = a.ks[-1]
    print(f"\n# zero-response check at K={kmax}: share of integrals that are exactly 0")
    for b in POT_BINS:
        p, n = rows[b][kmax]
        if not len(p) or not len(n):
            continue
        zp = float(np.mean(np.asarray(p, float) == 0)); zn = float(np.mean(np.asarray(n, float) == 0))
        note = "  <- empty integrand on both sides" if zp == 1.0 and zn == 1.0 else ""
        print(f"  bin {str(b):10} on-track {zp:6.1%} zero (n={len(p)}),  decoy {zn:6.1%} zero (n={len(n)}){note}")

    print("\n# a positive result needs AUC to rise with K by >=0.05 and exceed 0.62 in the POT<16 bins")
    any_go = False
    for b, a1, ak in go:
        ok = (ak - a1 >= 0.05) and (ak > 0.62)
        any_go = any_go or ok
        print(f"  bin {b}: K=1 {a1:.3f} -> K={a.ks[-1]} {ak:.3f}  (delta {ak - a1:+.3f})  -> {'GO' if ok else 'weak/NO'}")
    print(f"\n>>> verdict: {'signal found' if any_go else 'no signal: integration does not beat the floor -> FurrowMap counting'}")
    print("    (add clips to increase n; plant-vs-weed would need a separate test)")


if __name__ == "__main__":
    main()
