#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do luong chi tiet ma mot cam bien THUC SU giao ra, doc lap voi moi detector.

Bai khang dinh khung 8K chi mang lop chi tiet tuong duong 2560-3840 px. Phep do nay la
mot trong hai tru chong lung cho khang dinh do (tru kia la dao dau cua cat o), va no
khong dung toi mo hinh nao.

CAI BAY: so sanh tai "0.75 Nyquist" cua TUNG anh la KHONG hop le giua hai camera.
Nyquist cua moi anh phu thuoc mat do lay mau cua chinh no, nen 0.75*f_Nyq cua khung 8K
ung voi mot tan so GOC cao gap ~3.9 lan so voi khung 1080p (510 vs 1969 chu ky/rad).
Ket qua "8K it cong suat hon" la DUONG NHIEN, bat ke quang hoc ra sao.

Vi vay script co hai che do:
  --mode nyquist  ti so tai 0.75/0.25 Nyquist RIENG tung anh   (tai lap con so cu)
  --mode angular  ti so tai cung mot tan so GOC cho moi camera (phep so sanh hop le),
                  quy doi qua f/p: r_norm = 2*nu/(f/p), voi nu tinh bang chu ky/radian

Giao thuc, dung nhu Muc 2.5 cua bai mo ta:
  - lay N khung moi bo (mac dinh 8)
  - cat o trung tam 1024x1024
  - tinh pho cong suat trung binh theo ban kinh (radially-averaged)
  - chuan hoa theo gia tri o tan thap
  - bao cao ti so cong suat tai 0.75 Nyquist so voi tai 0.25 Nyquist

He so chuan hoa triet tieu trong ti so, nen no chi de doc duong cong cho de; con so
bao cao la P(0.75 f_Nyq) / P(0.25 f_Nyq).

Cam bien giao du chi tiet o buoc diem anh danh nghia thi con giu cong suat gan Nyquist;
cam bien co khung da noi suy, khu nhieu hay nen qua muc noi dung quang hoc thi khong.

  "$PY" exp_power_spectrum.py \\
      --dir "1080p=data/test/images" \\
      --dir "8K=/media/manhhv/DATA/AI/_archive/paper2/test2/images" \\
      --out results/crosssite/power_spectrum.csv

Chay voi --selftest de kiem tren anh tong hop truoc khi tin so tren anh that.
"""
import argparse
import csv
import glob
import os

import numpy as np


def radial_profile(crop, window="none"):
    """Pho cong suat trung binh theo ban kinh. Tra ve (r_norm, power) voi r_norm=1 tai Nyquist."""
    g = crop.astype(np.float64)
    g = g - g.mean()
    if window == "hann":
        n = g.shape[0]
        w = np.hanning(n)
        g = g * np.outer(w, w)
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    n = P.shape[0]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / (n / 2.0)   # 1.0 = Nyquist tren truc
    nb = n // 2
    idx = np.clip((r * nb).astype(int), 0, nb - 1)
    tot = np.bincount(idx.ravel(), weights=P.ravel(), minlength=nb)
    cnt = np.bincount(idx.ravel(), minlength=nb)
    prof = tot / np.maximum(cnt, 1)
    return (np.arange(nb) + 0.5) / nb, prof


def ratio_at(rn, prof, f_hi, f_lo):
    """P(f_hi) / P(f_lo), noi suy tuyen tinh tren luoi ban kinh."""
    return float(np.interp(f_hi, rn, prof) / max(np.interp(f_lo, rn, prof), 1e-30))


def central_crop(im, size):
    h, w = im.shape[:2]
    if min(h, w) < size:
        return None
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return im[y0:y0 + size, x0:x0 + size]


def measure(images_dir, n_frames, size, f_hi, f_lo, window, fp=None, mode="nyquist"):
    """f_hi/f_lo la PHAN cua Nyquist khi mode=nyquist; la chu ky/rad khi mode=angular."""
    import cv2
    files = sorted(sum((glob.glob(os.path.join(images_dir, e))
                        for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG")), []))
    if not files:
        raise SystemExit(f"khong co anh trong {images_dir}")
    step = max(1, len(files) // n_frames)          # trai deu, khong lay 8 khung lien tiep
    picks = files[::step][:n_frames]
    out = []
    for p in picks:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        c = central_crop(im, size)
        if c is None:
            continue
        rn, prof = radial_profile(c, window)
        if mode == "angular":
            hi, lo = 2.0 * f_hi / fp, 2.0 * f_lo / fp    # chu ky/rad -> phan cua Nyquist
        else:
            hi, lo = f_hi, f_lo
        out.append((os.path.basename(p), ratio_at(rn, prof, hi, lo), im.shape))
    return out


def selftest():
    """Anh mo (loc thap) phai cho ti so NHO hon anh sac net. Kiem huong, khong kiem gia tri."""
    import cv2
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (1024, 1024), dtype=np.uint8)
    blur = cv2.GaussianBlur(sharp, (0, 0), 2.0)
    rs = ratio_at(*radial_profile(sharp), 0.75, 0.25)
    rb = ratio_at(*radial_profile(blur), 0.75, 0.25)
    print(f"  nhieu trang (sac net) : {rs:.4f}")
    print(f"  sau khi lam mo sigma=2: {rb:.6f}")
    assert rb < rs / 10, "tu kiem THAT BAI: lam mo phai keo ti so xuong manh"
    print("  [OK] tu kiem dat: mat chi tiet -> ti so tut.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=[],
                    help="NHAN=duong/dan[=f_p], lap lai cho tung bo; f_p bat buoc voi --mode angular")
    ap.add_argument("--mode", choices=("nyquist", "angular"), default="nyquist")
    ap.add_argument("--nu-hi", type=float, default=510.0, help="chu ky/rad, tu so (che do angular)")
    ap.add_argument("--nu-lo", type=float, default=170.0, help="chu ky/rad, mau so (che do angular)")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--f-hi", type=float, default=0.75, help="phan cua Nyquist, tu so")
    ap.add_argument("--f-lo", type=float, default=0.25, help="phan cua Nyquist, mau so")
    ap.add_argument("--window", choices=("none", "hann"), default="none",
                    help="bai khong neu cua so; 'hann' de kiem do nhay")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    selftest()
    if a.selftest or not a.dir:
        return

    rows = []
    for spec in a.dir:
        parts = spec.split("=")
        label, path = parts[0], parts[1]
        fp = float(parts[2]) if len(parts) > 2 else None
        if a.mode == "angular" and fp is None:
            raise SystemExit(f"--mode angular can f/p: dung NHAN=duong/dan=F_P cho '{label}'")
        hi, lo = (a.nu_hi, a.nu_lo) if a.mode == "angular" else (a.f_hi, a.f_lo)
        res = measure(path, a.n_frames, a.crop, hi, lo, a.window, fp, a.mode)
        vals = np.array([r[1] for r in res])
        print(f"{label}  ({len(res)} khung, {res[0][2][1]}x{res[0][2][0]}, cat {a.crop}^2, window={a.window})")
        unit = "cyc/rad" if a.mode == "angular" else "Nyq"
        print(f"  P({hi} {unit})/P({lo} {unit}) = {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}"
              f"   [min {vals.min():.4f}, max {vals.max():.4f}]")
        for name, v, _ in res:
            rows.append([label, a.mode, name, f"{v:.6f}"])
    if len(a.dir) == 2:
        m = [np.mean([float(r[3]) for r in rows if r[0] == s.split("=")[0]]) for s in a.dir]
        print(f"\nti le giua hai bo: {max(m)/min(m):.1f} lan")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["corpus", "mode", "frame", "ratio"])
            w.writerows(rows)
        print(f"[OK] -> {a.out}")


if __name__ == "__main__":
    main()
