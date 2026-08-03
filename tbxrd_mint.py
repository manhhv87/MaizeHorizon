#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track-back minting: optical-flow pseudo-labels for the far tier (TB-XRD stage 1).

The detector cannot see distant plants, so far boxes cannot come from it directly. Instead:

  1. Forward-track with ByteTrack and confirm a real approaching plant: the track needs at least
     --min-anchors confident near detections (conf >= --conf-anchor, POT >= --t-near) of
     monotonically growing size. The detector is reliable here because near plants are large.
  2. Seed = the earliest confident detection of that track.
  3. Track backwards from the seed with Lucas-Kanade optical flow, following real pixels rather
     than the detector or a linear extrapolation. The chain stops when the plant gets too small,
     leaves the frame, loses features, or loses its green signature. Each step is guarded by a
     forward-backward check and a green gate.
  4. Backward boxes below --t-near become far labels, snapped to the dominant green blob and
     deduplicated by NMS.
"""
import argparse, os, glob, json, csv, re
from collections import defaultdict
import numpy as np

GREEN_LO = (35, 40, 40)
GREEN_HI = (85, 255, 255)


def frame_key(p):
    """Natural sort by frame number, so the order is temporal regardless of zero padding."""
    m = re.findall(r"\d+", os.path.basename(p))
    return (int(m[-1]) if m else -1, os.path.basename(p))


# ----------------------------- helpers -----------------------------
def green_mask(bgr):
    import cv2
    return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), GREEN_LO, GREEN_HI)


def green_frac(bgr):
    m = green_mask(bgr)
    return float((m > 0).mean()) if m.size else 0.0


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def box_of(s):
    _, cx, cyb, w, h, _, _ = s
    return [cx - w / 2.0, cyb - h, cx + w / 2.0, cyb]


def clampi(box, W, H):
    return [int(max(box[0], 0)), int(max(box[1], 0)), int(min(box[2], W)), int(min(box[3], H))]


def refine_box_to_plant(im, box, args):
    """Snap a box to the tight bbox of the green region inside the local ROI."""
    import cv2
    H, W = im.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(round(bw * args.roi_margin)), int(round(bh * args.roi_margin))
    rx1, ry1 = max(int(x1 - mx), 0), max(int(y1 - my), 0)
    rx2, ry2 = min(int(x2 + mx), W), min(int(y2 + my), H)
    roi = im[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    mask = (green_mask(roi) > 0).astype(np.uint8)
    if int(mask.sum()) < 4:
        return None
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    ob = (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1)
    best, best_inter = None, 0
    for i in range(1, n):
        L, T = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        Wc, Hc = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        inter = max(0, min(L + Wc, ob[2]) - max(L, ob[0])) * max(0, min(T + Hc, ob[3]) - max(T, ob[1]))
        if inter > best_inter:
            best_inter, best = inter, i
    if best is None:
        return None
    L, T = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP]
    Wc, Hc = stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    if Wc * Hc > args.max_fill * roi.shape[0] * roi.shape[1]:
        return None
    fx1, fy1 = rx1 + L, ry1 + T
    if Hc < args.min_px or Wc < 2:
        return None
    return [int(fx1), int(fy1), int(fx1 + Wc), int(fy1 + Hc)]


def _inter(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _same_plant(a, b):
    """Same plant: strong overlap in x and some overlap in y, so two plants along the row are not merged."""
    xo = min(a[2], b[2]) - max(a[0], b[0])
    yo = min(a[3], b[3]) - max(a[1], b[1])
    if xo <= 0 or yo <= 0:
        return False
    wmin = max(min(a[2] - a[0], b[2] - b[0]), 1)
    hmin = max(min(a[3] - a[1], b[3] - b[1]), 1)
    return (xo / wmin > 0.6) and (yo / hmin > 0.6)   # chi gop box GAN TRUNG, KHONG gop 2 cay xa khac nhau cung cot


def nms(cands, iou_thr, ios_thr):
    """Merge duplicates by IoU, intersection-over-smaller, or the same-plant test."""
    cs = sorted(cands, key=lambda c: (c["green"],
              (c["xyxy"][2] - c["xyxy"][0]) * (c["xyxy"][3] - c["xyxy"][1])), reverse=True)
    keep = []
    for c in cs:
        a = c["xyxy"]; aa = (a[2] - a[0]) * (a[3] - a[1]); dup = False
        for k in keep:
            b = k["xyxy"]; ii = _inter(a, b)
            if ii > 0:
                ab = (b[2] - b[0]) * (b[3] - b[1])
                if ii / (aa + ab - ii) > iou_thr or ii / max(min(aa, ab), 1) > ios_thr:
                    dup = True; break
            if _same_plant(a, b):
                dup = True; break
        if not dup:
            keep.append(c)
    return keep


# ----------------------------- backward optical-flow tracker -----------------------------
def track_back_lk(frames, seed_idx, seed_box, args):
    """Track backwards from the seed with Lucas-Kanade and a forward-backward check."""
    import cv2
    cur = cv2.imread(frames[seed_idx])
    if cur is None:
        return []
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    H, W = cur_gray.shape
    box = [float(v) for v in seed_box]
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    out = []
    for k in range(1, args.back + 1):
        t = seed_idx - k
        if t < 0:
            break
        prev = cv2.imread(frames[t])
        if prev is None:
            break
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        cb = clampi(box, W, H)
        if cb[2] - cb[0] < 4 or cb[3] - cb[1] < 4:
            break
        mask = np.zeros((H, W), np.uint8); mask[cb[1]:cb[3], cb[0]:cb[2]] = 255
        p0 = cv2.goodFeaturesToTrack(cur_gray, maxCorners=120, qualityLevel=0.01, minDistance=3, mask=mask)
        if p0 is None or len(p0) < 6:
            break
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p0, None, **lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p1, None, **lk)
        if p1 is None or p0r is None:
            break
        a0 = p0.reshape(-1, 2); a1 = p1.reshape(-1, 2); ar = p0r.reshape(-1, 2)
        fb = np.linalg.norm(a0 - ar, axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 2.0)
        g0, g1 = a0[good], a1[good]
        if len(g1) < 6:
            break
        d = np.median(g1 - g0, axis=0)                          # tinh tien
        c0, c1 = g0.mean(0), g1.mean(0)
        s0 = np.median(np.linalg.norm(g0 - c0, axis=1)); s1 = np.median(np.linalg.norm(g1 - c1, axis=1))
        sc = float(np.clip(s1 / max(s0, 1e-3), 0.5, 1.0))       # lui -> CHI nho dan (<=1, khong cho phinh)
        cx = (box[0] + box[2]) / 2 + d[0]; cy = (box[1] + box[3]) / 2 + d[1]
        w = (box[2] - box[0]) * sc; h = (box[3] - box[1]) * sc
        nb = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        if h < args.min_px or w < 2 or not (0 <= cx <= W and 0 <= cy <= H):
            break
        bx = clampi(nb, W, H)
        crop = prev[bx[1]:bx[3], bx[0]:bx[2]]
        if crop.size == 0 or green_frac(crop) < args.green_min:  # mat cay (bi che / trat) -> dung
            break
        out.append((t, nb))
        box = nb; cur_gray = prev_gray
    return out


# ----------------------------- pass 1: track -----------------------------
def track_and_detect(model, frames, args):
    import cv2
    tracks, frame_dets, HW = {}, {}, None
    for idx, fp in enumerate(frames):
        im = cv2.imread(fp)
        if im is None:
            continue
        H, W = im.shape[:2]; HW = (H, W)
        r = model.track(im, persist=True, tracker="bytetrack.yaml", conf=args.conf,
                        imgsz=args.imgsz, classes=[args.plant_class], device=args.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            frame_dets[idx] = []
            continue
        xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        frame_dets[idx] = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)) for b, c in zip(xy, cf)]
        if r.boxes.id is None:
            continue
        ids = r.boxes.id.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), tid, c in zip(xy, ids, cf):
            w = x2 - x1; h = y2 - y1; pot = (w * h) / (W * H)
            tracks.setdefault(int(tid), []).append((idx, (x1 + x2) / 2.0, float(y2), float(w), float(h), float(c), float(pot)))
    return tracks, frame_dets, HW


def confirm_track(seq, args):
    """Confirm a real plant and return (seed, teacher): earliest anchor and largest-POT anchor."""
    seq = sorted(seq)
    anchors = [s for s in seq if s[5] >= args.conf_anchor and s[6] >= args.t_near]
    if len(anchors) < args.min_anchors:
        return None, "few_anchors"
    f = np.array([s[0] for s in anchors], float)
    h_ = np.array([s[4] for s in anchors], float)
    if np.ptp(f) < 1:
        return None, "no_frame_span"
    mh, bh = np.linalg.lstsq(np.vstack([f, np.ones_like(f)]).T, h_, rcond=None)[0]
    if mh <= 0:
        return None, "not_growing"
    if float(np.sqrt(np.mean((h_ - (mh * f + bh)) ** 2))) > args.max_resid * max(np.median(h_), 1e-6):
        return None, "noisy_fit"
    return dict(seed=anchors[0], teacher=max(anchors, key=lambda s: s[6]), n_anchor=len(anchors)), "ok"


def mint_seed(tid, info, frames, HW, args, clip):
    import cv2
    H, W = HW
    teacher = info["teacher"]; t_box = box_of(teacher)
    teacher_meta = dict(frame=os.path.basename(frames[teacher[0]]), idx=int(teacher[0]),
                        xyxy=[int(round(v)) for v in t_box], pot=float(teacher[6]))
    seed = info["seed"]
    bw = track_back_lk(frames, int(seed[0]), box_of(seed), args)
    cands = []
    for t, nb in bw:
        pot = ((nb[2] - nb[0]) * (nb[3] - nb[1])) / (W * H)
        if pot >= args.t_near:                                  # chi mint tang far
            continue
        im = cv2.imread(frames[t])
        if im is None:
            continue
        rb = clampi(nb, W, H)
        if not args.no_refine:
            rr = refine_box_to_plant(im, rb, args)
            if rr is not None:
                rb = rr
        rpot = ((rb[2] - rb[0]) * (rb[3] - rb[1])) / (W * H)
        if rpot >= args.t_near:                              # refine co the phong to -> kiem LAI tang far
            continue
        rcrop = im[rb[1]:rb[3], rb[0]:rb[2]]
        if rcrop.size == 0:
            continue
        cands.append(dict(frame_idx=int(t), xyxy=rb, pot=float(rpot),
                          green=float(green_frac(rcrop)), track_id=int(tid), teacher=teacher_meta))
    return cands


# ----------------------------- outputs -----------------------------
def write_yolo(out_dir, frames, mints_by_frame, frame_dets, args):
    """Write full labels per minted frame: confident near detections plus minted far boxes, deduplicated."""
    import cv2
    pc = args.plant_class
    lab_dir = os.path.join(out_dir, "labels"); os.makedirs(lab_dir, exist_ok=True)
    H = W = None
    for idx, mints in mints_by_frame.items():
        if H is None:
            im = cv2.imread(frames[idx]); H, W = im.shape[:2]
        boxes = [(m["xyxy"], pc) for m in mints]               # minted far -> plant
        for b in frame_dets.get(idx, []):
            bb = [b[0], b[1], b[2], b[3]]
            if max((iou(bb, mb) for mb, _ in boxes), default=0.0) >= 0.4:
                continue
            if b[4] >= args.conf_anchor:
                boxes.append((bb, pc))                          # near chac chan -> plant
            else:
                boxes.append((bb, args.ignore_class))          # mid-conf [conf,anchor) -> IGNORE (KHONG coi la nen)
        stem = os.path.splitext(os.path.basename(frames[idx]))[0]
        lines = [f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"
                 for (x1, y1, x2, y2), cls in boxes]
        with open(os.path.join(lab_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))


def write_overlays(out_dir, frames, frame_dets, mints_by_frame, args):
    import cv2
    ov_dir = os.path.join(out_dir, "overlay"); os.makedirs(ov_dir, exist_ok=True)
    idxs = sorted(mints_by_frame.keys())
    if len(idxs) > args.limit_overlay:
        idxs = [idxs[i] for i in np.linspace(0, len(idxs) - 1, args.limit_overlay).astype(int)]
    for idx in idxs:
        im = cv2.imread(frames[idx])
        if im is None:
            continue
        for b in frame_dets.get(idx, []):
            if b[4] >= args.conf_anchor:
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 1)
        for m in mints_by_frame[idx]:
            x1, y1, x2, y2 = m["xyxy"]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(im, f"g{m['green']:.2f}", (x1, max(y1 - 3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1, cv2.LINE_AA)
        stem = os.path.splitext(os.path.basename(frames[idx]))[0]
        cv2.imwrite(os.path.join(ov_dir, stem + ".jpg"), im)
    return ov_dir


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence for the forward track, used to pick confident near seeds")
    ap.add_argument("--device", default="0")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1, help="class for mid-confidence boxes, so they are not taught as background")
    ap.add_argument("--conf-anchor", type=float, default=0.40)
    ap.add_argument("--t-near", type=float, default=0.004, help="POT threshold separating near/teacher from far")
    ap.add_argument("--min-anchors", type=int, default=4)
    ap.add_argument("--max-resid", type=float, default=0.6)
    ap.add_argument("--back", type=int, default=20, help="maximum number of frames to track back per seed")
    ap.add_argument("--min-px", type=float, default=6.0)
    ap.add_argument("--green-min", type=float, default=0.08)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--roi-margin", type=float, default=0.30)
    ap.add_argument("--max-fill", type=float, default=0.85)
    ap.add_argument("--iou-skip", type=float, default=0.30)
    ap.add_argument("--nms-iou", type=float, default=0.40)
    ap.add_argument("--ios-merge", type=float, default=0.50, help="merge nested boxes (intersection over smallerer)")
    ap.add_argument("--limit-overlay", type=int, default=80)
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    frames = sorted([p for e in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(args.clip_dir, e))],
                    key=frame_key)
    if not frames:
        raise SystemExit(f"No images in {args.clip_dir}")
    clip = os.path.basename(os.path.normpath(args.clip_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[i] clip={clip} | {len(frames)} frame | imgsz={args.imgsz} | back={args.back} (LK optical-flow)")
    model = YOLO(args.weights)

    tracks, frame_dets, HW = track_and_detect(model, frames, args)
    print(f"[i] {len(tracks)} track.")

    chains, rows = [], []
    n_pass = 0; reasons = {}
    for tid, seq in tracks.items():
        info, why = confirm_track(seq, args)
        reasons[why] = reasons.get(why, 0) + 1
        if info is None:
            rows.append(dict(track_id=tid, status=why, n_anchor=0, n_cand=0)); continue
        n_pass += 1
        cands = mint_seed(tid, info, frames, HW, args, clip)
        rows.append(dict(track_id=tid, status="confirmed", n_anchor=info["n_anchor"], n_cand=len(cands)))
        if cands:
            chains.append(cands)

    # ---- chain dedup: two chains overlapping over many frames are the same plant ----
    chains.sort(key=lambda ch: (len(ch), float(np.mean([c["green"] for c in ch]))), reverse=True)
    kept_maps, all_c, n_chain_drop = [], [], 0
    for ch in chains:
        cmap = {c["frame_idx"]: c["xyxy"] for c in ch}
        dup = False
        for km in kept_maps:
            shared = set(cmap) & set(km)
            if len(shared) < max(3, int(0.3 * min(len(cmap), len(km)))):
                continue                                        # bang chung yeu -> KHONG coi la trung chain
            if sum(1 for f in shared if iou(cmap[f], km[f]) > 0.5) / len(shared) > 0.6:
                dup = True; break
        if dup:
            n_chain_drop += 1; continue
        kept_maps.append(cmap); all_c.extend(ch)

    by_frame = defaultdict(list)
    for c in all_c:
        by_frame[c["frame_idx"]].append(c)
    mints_by_frame, all_pairs = {}, []
    for idx, cs in by_frame.items():
        hi = [b for b in frame_dets.get(idx, []) if b[4] >= args.conf_anchor]
        cs = [c for c in cs if max((iou(c["xyxy"], b) for b in hi), default=0.0) < args.iou_skip]
        kept = nms(cs, args.nms_iou, args.ios_merge)
        if not kept:
            continue
        mints_by_frame[idx] = kept
        for c in kept:
            all_pairs.append(dict(clip=clip, track_id=c["track_id"], plant_uid=f"{clip}_t{c['track_id']}",
                far=dict(frame=os.path.basename(frames[idx]), idx=int(idx), xyxy=c["xyxy"], pot=c["pot"], green=c["green"]),
                near=c["teacher"]))

    write_yolo(args.out_dir, frames, mints_by_frame, frame_dets, args)
    with open(os.path.join(args.out_dir, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "mint_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "status", "n_anchor", "n_cand"])
        w.writeheader(); [w.writerow(r) for r in rows]
    ov_dir = write_overlays(args.out_dir, frames, frame_dets, mints_by_frame, args) if (not args.no_overlay and mints_by_frame) else None

    n_mint = sum(len(v) for v in mints_by_frame.values())
    pots = np.array([m["pot"] * 100 for v in mints_by_frame.values() for m in v]) if n_mint else np.array([])
    grns = np.array([m["green"] for v in mints_by_frame.values() for m in v]) if n_mint else np.array([])
    chain = np.array([r["n_cand"] for r in rows if r["status"] == "confirmed"]) if n_pass else np.array([])
    print("\n================ minting results (LK backward) ================")
    print(f"tracks confirmed as real plants: {n_pass}/{len(tracks)}  | rejected: "
          + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()) if k != 'ok'))
    print(f"duplicate chains merged: {n_chain_drop}  ->  {len(kept_maps)} independent chains")
    if len(chain):
        print(f"back-track length per seed: median={np.median(chain):.1f}  max={chain.max()}")
    print(f"candidates before dedup: {len(all_c)}  ->  minted after dedup: {n_mint} over {len(mints_by_frame)} frames "
          f"| cap near/far: {len(all_pairs)}")
    if n_mint:
        print(f"POT minted (%): median={np.median(pots):.3f}  p25={np.percentile(pots,25):.3f}  p75={np.percentile(pots,75):.3f}")
        print(f"do-xanh minted: median={np.median(grns):.3f}  min={grns.min():.3f}")
    print(f"\n-> labels: {os.path.join(args.out_dir,'labels')} | pairs: pairs.jsonl"
          + (f" | overlay: {ov_dir}" if ov_dir else ""))


if __name__ == "__main__":
    main()
