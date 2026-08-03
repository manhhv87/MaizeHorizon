#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a ground-truth ledger of distinct plants from sparse per-frame hand labels.

Hand labels cover only about 40 frames per clip. This links boxes of the same physical plant
across them: from each labelled frame, every box is LK-tracked forward through the dense frames to
the next labelled frame and matched there by IoU, forming a tracklet. One tracklet is one plant.
Writes a JSON ledger plus overlays for human verification.

Matching happens at labelled frames only, so the ledger never uses the tracker being evaluated.

  python furrowmap_ledger.py --frames-dir data/images/IMG_3916_test \
      --gt-dir data/test/labels --out results/counting/ledger_IMG_3916.json --vis-dir _ledger_vis/IMG_3916
"""
import argparse
import glob
import json
import os

import numpy as np

try:
    from tbxrd_mint import clampi
except Exception:                      # fallback neu import loi
    def clampi(b, W, H):
        return [max(0, min(W - 1, int(b[0]))), max(0, min(H - 1, int(b[1]))),
                max(1, min(W, int(b[2]))), max(1, min(H, int(b[3])))]


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def read_plants(lp, W, H, plant_cls=0):
    out = []
    if not os.path.exists(lp):
        return out
    for line in open(lp, encoding="utf-8"):
        s = line.split()
        if len(s) < 5 or int(float(s[0])) != plant_cls:
            continue
        xc, yc, bw, bh = map(float, s[1:5])
        out.append([(xc - bw / 2) * W, (yc - bh / 2) * H, (xc + bw / 2) * W, (yc + bh / 2) * H])
    return out


def lk_propagate(frames, i0, i1, box):
    """Track a box forward from frame i0 to i1 through the dense frames; returns the final box."""
    import cv2
    cur = cv2.imread(frames[i0])
    if cur is None:
        return None
    cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY); H, W = cur_g.shape
    b = [float(v) for v in box]
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    for t in range(i0 + 1, i1 + 1):
        nxt = cv2.imread(frames[t])
        if nxt is None:
            return None
        nxt_g = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
        cb = clampi(b, W, H)
        if cb[2] - cb[0] < 4 or cb[3] - cb[1] < 4:
            return None
        mask = np.zeros((H, W), np.uint8); mask[cb[1]:cb[3], cb[0]:cb[2]] = 255
        p0 = cv2.goodFeaturesToTrack(cur_g, maxCorners=120, qualityLevel=0.01, minDistance=3, mask=mask)
        if p0 is None or len(p0) < 6:
            return None
        p1, s1, _ = cv2.calcOpticalFlowPyrLK(cur_g, nxt_g, p0, None, **lk)
        p0r, s2, _ = cv2.calcOpticalFlowPyrLK(nxt_g, cur_g, p1, None, **lk)
        if p1 is None or p0r is None:
            return None
        a0 = p0.reshape(-1, 2); a1 = p1.reshape(-1, 2); ar = p0r.reshape(-1, 2)
        good = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm(a0 - ar, axis=1) < 2.0)
        g0, g1 = a0[good], a1[good]
        if len(g1) < 6:
            return None
        d = np.median(g1 - g0, axis=0)
        c0, c1 = g0.mean(0), g1.mean(0)
        s0 = np.median(np.linalg.norm(g0 - c0, axis=1)); s1_ = np.median(np.linalg.norm(g1 - c1, axis=1))
        sc = float(np.clip(s1_ / max(s0, 1e-3), 0.8, 1.5))     # FORWARD -> cho phep phong to
        cx = (b[0] + b[2]) / 2 + d[0]; cy = (b[1] + b[3]) / 2 + d[1]
        w = (b[2] - b[0]) * sc; h = (b[3] - b[1]) * sc
        b = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]; cur_g = nxt_g
    return b


def merge_fragments(tracks, labeled_order, max_gap=2, tau=75.0, passes=3):
    """Rejoin tracklets broken by LK: B starts just after A ends and sits where A's motion predicts.
    Greedy, multiple passes; returns how many tracklets were merged."""
    pos = {f: i for i, f in enumerate(labeled_order)}

    def stems(tr):
        return [(fi, (b[0] + b[2]) / 2.0, b[3]) for fi, b in tr["hist"]]

    total = 0
    for _ in range(passes):
        order = sorted(tracks, key=lambda k: tracks[k]["hist"][0][0])
        merged = set()
        for bid in order:
            if bid in merged or bid not in tracks:
                continue
            B = tracks[bid]; bf = B["hist"][0][0]
            if bf not in pos:
                continue
            bsx = (B["hist"][0][1][0] + B["hist"][0][1][2]) / 2.0; bsy = B["hist"][0][1][3]
            best, bestd = None, tau
            for aid in tracks:
                if aid == bid or aid in merged:
                    continue
                A = tracks[aid]; af = A["hist"][-1][0]
                if af not in pos or pos[bf] - pos[af] < 1 or pos[bf] - pos[af] > max_gap:
                    continue
                ast = stems(A)
                if len(ast) >= 2:
                    (f1, x1, y1), (f2, x2, y2) = ast[-2], ast[-1]
                    ds = max(1, pos[f2] - pos[f1]); steps = pos[bf] - pos[f2]
                    px = x2 + (x2 - x1) / ds * steps; py = y2 + (y2 - y1) / ds * steps
                else:
                    px, py = ast[-1][1], ast[-1][2]
                d = ((px - bsx) ** 2 + (py - bsy) ** 2) ** 0.5
                if d < bestd:
                    bestd, best = d, aid
            if best is not None:
                tracks[best]["hist"].extend(B["hist"]); tracks[best]["hist"].sort()
                merged.add(bid)
        for m in merged:
            tracks.pop(m, None)
        total += len(merged)
        if not merged:
            break
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="frame DAY (to_label/<clip>)")
    ap.add_argument("--gt-dir", required=True, help="directory of hand-labelled *.txt files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vis-dir", default=None, help="optional overlay output for human verification")
    ap.add_argument("--match-iou", type=float, default=0.2)
    ap.add_argument("--min-frames", type=int, default=2, help="a plant must appear in at least this many labelled frames")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1280, help="used to compute POT = box_h * imgsz/max(W,H)")
    ap.add_argument("--min-pot-reach", type=float, default=16.0,
                    help="keep only plants that reach this POT, i.e. that pass the near zone. "
                         "Cay far-only (khong bao gio toi gan) -> bo khoi GT (mo hinh quay-tinh-tien).")
    ap.add_argument("--merge-gap", type=int, default=2, help="fragment merging: maximum gap in labelled frames between A and B")
    ap.add_argument("--merge-tau", type=float, default=75.0, help="fragment merging: distance threshold (px) when ngoai suy")
    ap.add_argument("--count-line", type=float, default=0.72, help="vi tri vach dem (ty le chieu cao, 0..1)")
    a = ap.parse_args()
    import cv2

    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")) +
                    glob.glob(os.path.join(a.frames_dir, "*.png")) +
                    glob.glob(os.path.join(a.frames_dir, "*.jpeg")))
    if not frames:
        raise SystemExit(f"No frames found in {a.frames_dir}")
    stem2idx = {os.path.splitext(os.path.basename(f))[0]: i for i, f in enumerate(frames)}
    im0 = cv2.imread(frames[0]); H, W = im0.shape[:2]

    # labelled frames, in dense order
    labeled = []
    for lp in glob.glob(os.path.join(a.gt_dir, "*.txt")):
        stem = os.path.splitext(os.path.basename(lp))[0]
        if stem in stem2idx:
            labeled.append((stem2idx[stem], read_plants(lp, W, H, a.plant_class)))
    labeled.sort()
    if len(labeled) < 2:
        raise SystemExit(f"Need >=2 labelled frames matching the frame dir; found {len(labeled)}.")
    print(f"[i] {len(frames)} dense frames | {len(labeled)} labelled | {sum(len(g) for _,g in labeled)} GT boxes")

    # seed tracklets from the first labelled frame
    nid = 0
    tracks = {}     # id -> {"hist": [(frame_idx, box)], "active": bool, "box": last_box, "frame": last_labeled_idx}
    for box in labeled[0][1]:
        tracks[nid] = {"hist": [(labeled[0][0], box)], "active": True, "box": box, "frame": labeled[0][0]}; nid += 1

    for (la, _), (lb, gtb) in zip(labeled[:-1], labeled[1:]):
        # 1) propagate active tracklets to the next labelled frame
        preds = {}
        for tid, tr in tracks.items():
            if not tr["active"] or tr["frame"] != la:
                continue
            pb = lk_propagate(frames, la, lb, tr["box"])
            if pb is not None:
                preds[tid] = pb
        # 2) greedy IoU match against GT boxes at lb
        used_gt = set()
        pairs = sorted(((iou(preds[tid], gtb[j]), tid, j) for tid in preds for j in range(len(gtb))
                        if iou(preds[tid], gtb[j]) >= a.match_iou), reverse=True)
        used_t = set()
        for _, tid, j in pairs:
            if tid in used_t or j in used_gt:
                continue
            used_t.add(tid); used_gt.add(j)
            tracks[tid]["hist"].append((lb, gtb[j])); tracks[tid]["box"] = gtb[j]; tracks[tid]["frame"] = lb
        # 3) unmatched tracklets end; unmatched GT boxes start new ones
        for tid in tracks:
            if tracks[tid]["active"] and tracks[tid]["frame"] == la and tid not in used_t:
                tracks[tid]["active"] = False
        for j in range(len(gtb)):
            if j not in used_gt:
                tracks[nid] = {"hist": [(lb, gtb[j])], "active": True, "box": gtb[j], "frame": lb}; nid += 1

    # rejoin tracklets broken by LK, to reduce over-counting before review
    labeled_order = [idx for idx, _ in labeled]
    n_merged = merge_fragments(tracks, labeled_order, max_gap=a.merge_gap, tau=a.merge_tau)
    print(f"[i] merged {n_merged} tracklets broken by LK, reducing over-count.")

    # finalize and keep only plants that reach the near zone
    scale = a.imgsz / max(W, H)
    confirmed = []; n_far_only = 0
    for tid, tr in tracks.items():
        if len(tr["hist"]) < a.min_frames:
            continue
        max_pot = max((b[3] - b[1]) * scale for _, b in tr["hist"])
        if max_pot < a.min_pot_reach:                # far-only: khong bao gio toi gan -> bo khoi GT dem
            n_far_only += 1; continue
        stems = [[int(fi), float((b[0] + b[2]) / 2), float(b[3])] for fi, b in tr["hist"]]   # (frame, cx, y2)
        boxes = [[int(fi), round(float(b[0]), 1), round(float(b[1]), 1), round(float(b[2]), 1),
                  round(float(b[3]), 1)] for fi, b in tr["hist"]]                            # (frame, x1,y1,x2,y2)
        rep = stems[len(stems) // 2]
        confirmed.append({"id": int(tid), "n_obs": len(stems), "max_pot": round(float(max_pot), 1),
                          "frames": [s[0] for s in stems], "stems": stems, "boxes": boxes, "rep_stem": rep[1:]})
    ledger = {"clip": os.path.basename(a.frames_dir.rstrip("/\\")), "n_frames": len(frames),
              "n_labeled": len(labeled), "match_iou": a.match_iou, "min_frames": a.min_frames,
              "imgsz": a.imgsz, "min_pot_reach": a.min_pot_reach,
              "n_plants": len(confirmed), "n_far_only_dropped": n_far_only, "plants": confirmed}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=1)
    print(f"[OK] LEDGER: {len(confirmed)} countable plants (POT>={a.min_pot_reach:.0f}px, >={a.min_frames} frames) -> {a.out}")
    print(f"    raw tracklets={nid}; dropped {n_far_only} far-only plants that never come near")
    print("    >>> review the overlays and correct n_plants if LK split or merged a track.")

    # overlays for review: box, stem dot and id
    if a.vis_dir:
        os.makedirs(a.vis_dir, exist_ok=True)
        PALETTE = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 200, 128), (200, 0, 128), (128, 200, 0)]
        cidlist = sorted(p["id"] for p in confirmed)
        colors = {tid: PALETTE[k % len(PALETTE)] for k, tid in enumerate(cidlist)}
        th = max(3, W // 500)                                   # do day theo do phan giai
        fs = max(0.8, W / 1280.0)                               # co chu
        ft = max(2, W // 700)                                   # do day chu
        rad = max(6, W // 240)                                  # ban kinh cham goc-than
        by_frame = {}
        for tid in cidlist:
            for (fi, b) in tracks[tid]["hist"]:
                by_frame.setdefault(fi, []).append((tid, b))
        cly = int(a.count_line * H)
        for fi, items in sorted(by_frame.items()):
            im = cv2.imread(frames[fi])
            if im is None:
                continue
            cv2.line(im, (0, cly), (W, cly), (255, 255, 255), max(2, W // 600))   # VACH DEM (trang, tranh trung mau box)
            cv2.putText(im, "VACH DEM", (10, cly - 8), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (255, 255, 255), ft)
            for tid, b in items:
                c = colors[tid]; bb = clampi(b, W, H)
                cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), c, th)            # box dam
                sx, sy = int((b[0] + b[2]) / 2), int(b[3])                          # goc-than (day-giua)
                cv2.circle(im, (sx, min(sy, H - 1)), rad, c, -1)                    # cham goc-than
                cv2.circle(im, (sx, min(sy, H - 1)), rad, (255, 255, 255), 2)
                lab = str(tid)
                (tw, tht), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
                ly = max(tht + 6, bb[1])
                cv2.rectangle(im, (bb[0], ly - tht - 8), (bb[0] + tw + 8, ly + 2), c, -1)   # nen chu
                cv2.putText(im, lab, (bb[0] + 4, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft)
            hdr = f"frame {fi} | {len(items)} cay (cung mau=cung cay)"
            cv2.rectangle(im, (0, 0), (min(W, int(20 * fs) + len(hdr) * int(13 * fs)), int(40 * fs)), (0, 0, 0), -1)
            cv2.putText(im, hdr, (8, int(28 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft)
            cv2.imwrite(os.path.join(a.vis_dir, f"f{fi:05d}.jpg"), im)
        print(f"    overlays -> {a.vis_dir}/ (same colour = same plant)")


if __name__ == "__main__":
    main()
