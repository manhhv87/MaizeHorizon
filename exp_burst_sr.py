#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-frame burst super-resolution on the far tier.

Forward motion gives several sub-pixel-shifted views of the same plant, so this is the one route
that could add real information rather than reweighting what a single frame already has:
track-before-detect integrates a detector score that is already near zero, whereas burst SR fuses
raw pixels and could in principle recover aliased high-frequency detail.

For each far ground-truth box: collect K sub-pixel-shifted crops by LK track-back, register them
with phase correlation, shift-and-add to an upscaled image, then measure high-frequency energy and
detector response. Two controls: bicubic upscaling of the single seed crop (no multi-frame
information), and a shuffled stack built from a different plant (wrong registration and content).
"""
import argparse
import csv
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # repo root
from tbxrd_mint import track_back_lk  # noqa: E402
from mve_tbd import read_corn, rank_auc  # noqa: E402
from rebuttal_common import load_detector  # noqa: E402


# ----------------------------- SR core (dependency-free) -----------------------------
def extract_context(im, box, ctx_scale=4.0):
    """Crop a square context window around a box; returns (patch, (ox, oy, side))."""
    H, W = im.shape[:2]
    x1, y1, x2, y2 = box[:4]
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    side = max((y2 - y1), (x2 - x1)) * ctx_scale
    side = int(max(round(side), 16))
    ox = int(round(cx - side / 2)); oy = int(round(cy - side / 2))
    ox = min(max(ox, 0), max(W - side, 0)); oy = min(max(oy, 0), max(H - side, 0))
    patch = im[oy:oy + side, ox:ox + side]
    if patch.shape[0] != side or patch.shape[1] != side:
        patch = im[oy:min(oy + side, H), ox:min(ox + side, W)]
    return patch, (ox, oy, side)


def register_shift(ref_gray, mov_gray):
    """Sub-pixel shift of mov relative to ref, by phase correlation."""
    import cv2
    h, w = ref_gray.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    r = ref_gray.astype(np.float32); m = cv2.resize(mov_gray, (w, h)).astype(np.float32)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(r, m, win)
    except Exception:  # noqa: BLE001
        dx, dy = 0.0, 0.0
    return float(dx), float(dy)


def shift_and_add_sr(lr_patches, shifts, scale):
    """Shift-and-add fusion of low-resolution patches into an upscaled image."""
    import cv2
    L = lr_patches[0].shape[0]
    Hh = Ww = int(L * scale)
    acc = np.zeros((Hh, Ww, 3), np.float32)
    wsum = np.zeros((Hh, Ww, 1), np.float32)
    ys, xs = np.mgrid[0:L, 0:L]
    for patch, (dx, dy) in zip(lr_patches, shifts):
        p = cv2.resize(patch, (L, L)).astype(np.float32)
        # toa do HR = (pixel - shift) * scale ; splat bilinear
        gx = (xs - dx) * scale; gy = (ys - dy) * scale
        x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int)
        fx = (gx - x0)[..., None]; fy = (gy - y0)[..., None]
        for (ax, ay, wgt) in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                              (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xx = x0 + ax; yy = y0 + ay
            m = (xx >= 0) & (xx < Ww) & (yy >= 0) & (yy < Hh)
            np.add.at(acc, (yy[m], xx[m]), (p * wgt)[m])
            np.add.at(wsum, (yy[m], xx[m]), wgt[m])
    hr = acc / np.maximum(wsum, 1e-6)
    holes = (wsum[..., 0] < 1e-3)
    if holes.any():                                   # lap lo bang bicubic cua khung dau
        base = cv2.resize(lr_patches[0], (Ww, Hh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        hr[holes] = base[holes]
    return np.clip(hr, 0, 255).astype(np.uint8)


def unsharp(im, amount=0.6, sigma=1.0):
    import cv2
    g = cv2.GaussianBlur(im, (0, 0), sigma)
    return np.clip(im.astype(np.float32) + amount * (im.astype(np.float32) - g), 0, 255).astype(np.uint8)


def highfreq_ratio(im):
    """High-frequency energy ratio: higher means more fine detail."""
    import cv2
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g)); mag = np.abs(F)
    h, w = g.shape; cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    hi = rad > 0.35 * min(h, w) / 2
    tot = mag.sum() + 1e-9
    return float(mag[hi].sum() / tot)


def realesrgan_sr(patch, scale):
    """Optional single-image learned SR (Real-ESRGAN) if installed."""
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import cv2
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        up = RealESRGANer(scale=4, model_path="RealESRGAN_x4plus.pth", model=model, half=False)
        out, _ = up.enhance(patch, outscale=scale)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[realesrgan] unavailable ({e}); using shift-and-add.")
        return None


# ----------------------------- detector on a patch -----------------------------
def detect_response(model, patch, gt_box_in_patch, imgsz, device, conf=0.02, plant_cls=0):
    """Return (max confidence over the patch, hit), hit = a plant box with IoU > 0.3 against GT."""
    import cv2
    r = model.predict(patch, conf=conf, iou=0.6, imgsz=imgsz, device=device,
                      classes=[plant_cls], verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return 0.0, 0
    xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
    gx1, gy1, gx2, gy2 = gt_box_in_patch
    best_conf = float(cf.max()) if len(cf) else 0.0
    hit = 0
    for (bx1, by1, bx2, by2), c in zip(xy, cf):
        ix1, iy1 = max(bx1, gx1), max(by1, gy1); ix2, iy2 = min(bx2, gx2), min(by2, gy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (bx2 - bx1) * (by2 - by1) + (gx2 - gx1) * (gy2 - gy1) - inter
        if ua > 0 and inter / ua > 0.3:
            hit = 1; break
    return best_conf, hit


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="one or more .pt files, comma separated (pooled over detector seeds)")
    ap.add_argument("--frames-dir", required=True, help="consecutive frames of one clip")
    ap.add_argument("--gt-dir", required=True, help="hand labels: anchor far GT and exclude maize from the shuffle control")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--scale", type=int, default=3, help="he so SR (x2..x4)")
    ap.add_argument("--k", type=int, default=5, help="maximum number of far frames to fuse")
    ap.add_argument("--pot-lo", type=float, default=6.0)
    ap.add_argument("--ctx-scale", type=float, default=4.0)
    ap.add_argument("--sr-backend", choices=["shiftadd", "realesrgan"], default="shiftadd")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--out", default="rebuttal_burst_sr.csv")
    a = ap.parse_args()

    tb_args = SimpleNamespace(back=a.k + 4, min_px=3.0, green_min=0.08)

    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")) +
                    glob.glob(os.path.join(a.frames_dir, "*.png")) +
                    glob.glob(os.path.join(a.frames_dir, "*.jpeg")))
    if len(frames) < 10:
        raise SystemExit(f"Can frame lien tiep (>=10); thay {len(frames)}")
    stem2idx = {os.path.splitext(os.path.basename(f))[0]: i for i, f in enumerate(frames)}
    im0 = cv2.imread(frames[0]); H, W = im0.shape[:2]
    scale_pot = a.imgsz / max(W, H)

    # gom far-GT seeds
    far_seeds = []
    for lp in glob.glob(os.path.join(a.gt_dir, "*.txt")):
        stem = os.path.splitext(os.path.basename(lp))[0]
        if stem not in stem2idx:
            continue
        idx = stem2idx[stem]
        corn = read_corn(lp, W, H, a.plant_class)
        for b in corn:
            pot = (b[3] - b[1]) * scale_pot
            if a.pot_lo <= pot < 16:
                far_seeds.append((idx, b))
    print(f"[i] {len(frames)} frames | {len(far_seeds)} far-GT seeds (POT {a.pot_lo:.0f}-16) | scale_pot={scale_pot:.3f}")
    if len(far_seeds) < 3:
        raise SystemExit("Too few far GT boxes; add clips or lower --pot-lo.")

    models = [load_detector(w, w) for w in a.weights.split(",") if os.path.exists(w)]
    if not models:
        raise SystemExit("No detector could be loaded.")

    # build one low-resolution context stack per seed (also the source for the shuffle control)
    stacks = []      # dict(seed, gt_in_patch_LR, lr_patches, side)
    for si, (idx, box) in enumerate(far_seeds):
        im = cv2.imread(frames[idx])
        if im is None:
            continue
        traj = [(idx, box)] + track_back_lk(frames, idx, box, tb_args)
        # giu cac khung ~cung ti le seed (chong bien dang scale khi hop nhat)
        seed_h = box[3] - box[1]
        sel = [(t, b) for (t, b) in traj if 0.7 * seed_h <= (b[3] - b[1]) <= 1.15 * seed_h][:a.k]
        if len(sel) < 2:
            continue
        # context window is fixed per seed
        _, (ox, oy, side) = extract_context(im, box, a.ctx_scale)
        lr = []
        for (t, b) in sel:
            imt = cv2.imread(frames[t])
            if imt is None:
                continue
            cx = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2
            ex = int(min(max(round(cx - side / 2), 0), max(W - side, 0)))
            ey = int(min(max(round(cy - side / 2), 0), max(H - side, 0)))
            crop = imt[ey:ey + side, ex:ex + side]
            if crop.shape[0] == side and crop.shape[1] == side:
                lr.append(crop)
        if len(lr) < 2:
            continue
        gt_lr = (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)
        stacks.append(dict(lr=lr, side=side, gt_lr=gt_lr))
    print(f"[i] {len(stacks)}/{len(far_seeds)} seeds have a stack of >=2 frames to fuse.")
    if len(stacks) < 3:
        raise SystemExit("Too few fusible stacks.")

    rng = np.random.default_rng(0)
    rows = [["seed", "cond", "hf_ratio", "mean_conf", "hit"]]
    agg = {c: {"conf": [], "hit": [], "hf": []} for c in ["bicubic", "burst", "shuffle"]}
    on_conf, sh_conf = [], []   # cho AUC(burst vs shuffle)

    for si, st in enumerate(stacks):
        lr = st["lr"]; side = st["side"]; sc = a.scale
        ref_gray = cv2.cvtColor(lr[0], cv2.COLOR_BGR2GRAY)
        shifts = [(0.0, 0.0)] + [register_shift(ref_gray, cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)) for p in lr[1:]]

        # --- burst SR ---
        if a.sr_backend == "realesrgan":
            burst = realesrgan_sr(lr[0], sc)
            if burst is None:
                burst = unsharp(shift_and_add_sr(lr, shifts, sc))
        else:
            burst = unsharp(shift_and_add_sr(lr, shifts, sc))
        # --- control A: bicubic 1 khung ---
        bicubic = cv2.resize(lr[0], (side * sc, side * sc), interpolation=cv2.INTER_CUBIC)
        # --- control B: shuffle (seed plus crops from a different plant) ---
        other = stacks[int(rng.integers(0, len(stacks)))]
        while other is st and len(stacks) > 1:
            other = stacks[int(rng.integers(0, len(stacks)))]
        oc = cv2.resize(other["lr"][0], (side, side))
        sh_stack = [lr[0], oc]
        sh_shift = [(0.0, 0.0), register_shift(ref_gray, cv2.cvtColor(oc, cv2.COLOR_BGR2GRAY))]
        shuffle = unsharp(shift_and_add_sr(sh_stack, sh_shift, sc))

        gt_hr = tuple(v * sc for v in st["gt_lr"])
        for cond, img in (("bicubic", bicubic), ("burst", burst), ("shuffle", shuffle)):
            hf = highfreq_ratio(img)
            confs, hits = [], []
            for model in models:
                c, hh = detect_response(model, img, gt_hr, a.imgsz, a.device, plant_cls=a.plant_class)
                confs.append(c); hits.append(hh)
            mc = float(np.mean(confs)); mh = float(np.mean(hits))
            rows.append([si, cond, round(hf, 5), round(mc, 4), round(mh, 3)])
            agg[cond]["conf"].append(mc); agg[cond]["hit"].append(mh); agg[cond]["hf"].append(hf)
            if cond == "burst":
                on_conf.append(mc)
            if cond == "shuffle":
                sh_conf.append(mc)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print("\n=== burst-SR results ({} detector seeds, {} far stacks) ===".format(len(models), len(stacks)))
    print(f"{'cond':10}{'hit_rate':>10}{'mean_conf':>11}{'hf_ratio':>10}")
    for c in ["bicubic", "burst", "shuffle"]:
        print(f"{c:10}{np.mean(agg[c]['hit']):>10.3f}{np.mean(agg[c]['conf']):>11.4f}{np.mean(agg[c]['hf']):>10.4f}")
    auc, npos = rank_auc(on_conf, sh_conf)
    dh_bic = np.mean(agg["burst"]["hit"]) - np.mean(agg["bicubic"]["hit"])
    dh_shf = np.mean(agg["burst"]["hit"]) - np.mean(agg["shuffle"]["hit"])
    dhf = np.mean(agg["burst"]["hf"]) - np.mean(agg["bicubic"]["hf"])
    print(f"\nAUC(burst-conf vs shuffle-conf) = {auc:.3f}  (>0.5 = burst tach khoi shuffle)")
    print(f"delta hit: burst-bicubic = {dh_bic:+.3f} | burst-shuffle = {dh_shf:+.3f} | delta hf(burst-bicubic) = {dhf:+.4f}")
    margin = 0.05
    go = (dh_bic >= margin) and (dh_shf >= margin) and (dhf > 0)
    print(f"\n>>> burst-SR verdict: "
          + ("GO -> burst-SR VUOT san single-frame (them chi tiet THAT). BAC BO gioi han single-frame; "
             "bao cao remedy #4 DUONG + cap nhat claim." if go else
             "NO-GO -> burst-SR KHONG vuot san (burst ~ bicubic ~ shuffle). Cung co claim; "
             "SIET ngon ngu paper: information limit KE CA voi hop nhat da-khung."))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
