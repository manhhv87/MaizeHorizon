#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teaser figure: one forward-facing frame with boxes coloured by POT tier.

near (>= 32 px) green, mid (16-32) orange, far (< 16) red; ignore regions shaded. Two insets show
one near and one far plant at the same display size, so the scale difference is visible.

No GPU needed: one frame and its YOLO label file are enough.

  python make_teaser.py --image <frame.jpg> --label <frame.txt> \\
      --out paper/figures/maizehorizon_example
  python make_teaser.py ... --lang vi --out paper/vi/figures/maizehorizon_example

Ve bang matplotlib chu khong bang OpenCV. Anh nen van la raster -- no la anh chup,
khong co cach nao khac -- nhung hop, chu, khung inset va duong noi nay la VECTOR, nen
phong to trong ban PDF khong bi rang cua. Ban truoc ve moi thu bang cv2 roi ghi JPEG,
tuc ca chu giai lan hop deu bi nen mat mat cung voi anh.

Ghi ra CA HAI dinh dang tu cung mot hinh: `.pdf` de nop bai va `.jpg` de xem nhanh.
`--out` nhan phan THAN ten, khong keo duoi.

Hai diem de sai:
  1. dpi phai chon sao cho 1 px anh = 1 px PDF, neu khong anh nen bi lay mau lai va
     mo di ma khong duoc gi. Kich thuoc hinh = (W/dpi, H/dpi).
  2. inset phong to bang NEAREST, khong noi suy: muc dich la cho thay cay xa that su
     con bao nhieu diem anh, noi suy se ve ra chi tiet khong ton tai.
"""
import argparse
import os

import numpy as np

from eval_testset import read_gt

IMGSZ = 1280
# Okabe--Ito: an toan cho nguoi mu mau (khong dung truc do--luc) va ba mau nay con
# khac nhau ve DO SANG, nen van tach duoc khi in den trang.
COL = {"near": "#009E73", "mid": "#E69F00", "far": "#0072B2"}
ACCENT = "#E6E600"          # khung inset + duong noi
LW = {"near": 1.4, "mid": 1.4, "far": 1.4}
# Nhan theo ngon ngu, cho ban tieng Viet cua paper (paper/vi). Mac dinh 'en'
# nen anh cua ban English khong doi.
L10N = {
    "en": {"near": "near (>=32px)", "mid": "mid (16-32px)", "far": "far (<16px)",
           "near view": "near view", "far view": "far view"},
    "vi": {"near": "gần (≥ 32 px)", "mid": "giữa (16–32 px)", "far": "xa (< 16 px)",
           "near view": "khung nhìn gần", "far view": "khung nhìn xa"},
}


def tier_of(box, w, h):
    pot = (box[3] - box[1]) * IMGSZ / max(w, h)
    return "near" if pot >= 32 else ("mid" if pot >= 16 else "far"), pot


def crop_box(box, W, H, pad_frac=0.6):
    """Khung cat quanh mot hop, noi rong pad_frac ve moi phia, ket o bien anh."""
    x1, y1, x2, y2 = box[:4]
    px, py = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    return (int(max(0, x1 - px)), int(max(0, y1 - py)),
            int(min(W, x2 + px)), int(min(H, y2 + py)))


def jpeg_images(pdf, quality=88):
    """Nen lai anh trong PDF thanh JPEG, giu nguyen phan vector.

    matplotlib nhung anh vao PDF duoi dang PNG khong mat mat. Voi mot ANH CHUP thi
    do la lang phi lon: khung 1920x1080 o day chiem 4,5 MB va lam ban thao phinh tu
    1,6 len 4,0 MB, doi lai khong duoc gi ma mat thuong khong thay. Ghostscript nen
    lai anh thanh DCT (JPEG) trong khi hop, chu va duong noi van la vector.

    Khong ha do phan giai (`-dDownsample*Images=false`): muc dich la giu du 1 px anh
    = 1 px PDF, chi doi kieu nen.

    Khong co ghostscript thi bo qua, chi bao mot dong -- file van dung, chi nang hon.
    """
    import shutil
    import subprocess
    gs = shutil.which("gs")
    if not gs:
        print("[!] khong co ghostscript, bo qua buoc nen anh")
        return
    tmp = pdf + ".tmp"
    cmd = [gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
           "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode",
           "-dAutoFilterGrayImages=false", "-dGrayImageFilter=/DCTEncode",
           "-dDownsampleColorImages=false", "-dDownsampleGrayImages=false",
           f"-dJPEGQ={quality}", "-dSubsetFonts=true", "-dEmbedAllFonts=true",
           f"-sOutputFile={tmp}", pdf]
    if subprocess.run(cmd, capture_output=True).returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, pdf)
    else:
        if os.path.exists(tmp):
            os.remove(tmp)
        print("[!] ghostscript that bai, giu ban PNG khong nen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="maizehorizon_example",
                    help="than ten, khong keo duoi: ghi ra <out>.pdf va <out>.jpg")
    ap.add_argument("--inset", type=int, default=220, help="canh inset (px anh)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--jpeg-quality", type=int, default=88,
                    help="chat luong JPEG cho anh nen trong PDF")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--lang", choices=("en", "vi"), default="en", help="ngon ngu nhan")
    a = ap.parse_args()
    T = L10N[a.lang]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, ConnectionPatch
    # Type 42 = TrueType nhung duoc. Mac dinh Type 3 bi nhieu nha xuat ban tu choi.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    img = plt.imread(a.image)
    if img is None or not getattr(img, "size", 0):
        raise SystemExit(f"could not read image: {a.image}")
    H, W = img.shape[:2]
    plants, ignores = read_gt(a.label, W, H, a.plant_class, a.ignore_class)
    if not plants:
        raise SystemExit(f"no plants in label file: {a.label}")

    fig = plt.figure(figsize=(W / a.dpi, H / a.dpi), dpi=a.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="none")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

    for b in ignores:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               facecolor="0.6", edgecolor="none", alpha=0.25, zorder=2))

    counts = {"near": 0, "mid": 0, "far": 0}
    for b in plants:
        t, _ = tier_of(b, W, H)
        counts[t] += 1
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                               edgecolor=COL[t], linewidth=LW[t], zorder=3))

    # chon 1 near (cao nhat) + 1 far (thap nhat) lam inset
    by_h = sorted(plants, key=lambda b: b[3] - b[1])
    far_b, near_b = by_h[0], by_h[-1]

    margin, s = 12, a.inset
    # Nen mo thay vi vien chu: `withStroke` ve chu thanh duong path, nen ban PDF mat
    # het chu chon duoc va nha xuat ban khong kiem duoc font. Hop nen giu chu that
    # ma van doc duoc tren anh.
    def box():
        return dict(facecolor="black", alpha=0.55, edgecolor="none",
                    boxstyle="round,pad=0.25")
    y0 = margin
    for b, key in ((near_b, "near view"), (far_b, "far view")):
        if y0 + s > H:
            break
        cx1, cy1, cx2, cy2 = crop_box(b, W, H)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        x0 = W - s - margin
        # Toa do hinh: truc chinh phu kin figure, nen s px = s/W ngang, s/H doc.
        iax = fig.add_axes([x0 / W, 1 - (y0 + s) / H, s / W, s / H], zorder=4)
        # NEAREST: cot loi cua hinh nay la cho thay cay xa con bao nhieu diem anh
        iax.imshow(img[cy1:cy2, cx1:cx2], interpolation="nearest", aspect="auto")
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_edgecolor("white"); sp.set_linewidth(1.6)
        iax.add_patch(Rectangle((b[0] - cx1, b[1] - cy1), b[2] - b[0], b[3] - b[1],
                                fill=False, edgecolor=ACCENT, linewidth=1.6))
        iax.text(0.04, 0.95, T[key], transform=iax.transAxes, ha="left", va="top",
                 fontsize=6, color="white", bbox=box())
        # duong noi tu canh trai inset toi chinh cay tren anh
        fig.add_artist(ConnectionPatch(
            xyA=(0, 0.5), coordsA=iax.transAxes,
            xyB=((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), coordsB=ax.transData,
            color=ACCENT, linewidth=0.8, zorder=3.5))
        y0 += s + margin

    # Moi tang mot dong RIENG de giu duoc mau cua tang do, va giang dong tinh bang
    # POINT chu khong bang px anh: co chu la point, nen giang dong theo px se chong
    # chu khi doi dpi. `offset points` giu hai thu cung mot don vi.
    fs = 6.5
    for i, t in enumerate(("near", "mid", "far")):
        # Neo lui vao 1 dong: hop nen cua dong dau se tran len khoi anh neu neo dung
        # sat mep, va bi cat mat mot phan trong ban dung.
        ax.annotate(f"{T[t]}: {counts[t]}", xy=(margin / W, 1 - margin / H),
                    xycoords="axes fraction", textcoords="offset points",
                    xytext=(0, -(i + 0.6) * fs * 1.9), ha="left", va="top",
                    fontsize=fs, color=COL[t], bbox=box(), zorder=5)

    stem = os.path.splitext(a.out)[0] if a.out.lower().endswith((".jpg", ".jpeg", ".png", ".pdf")) else a.out
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(f"{stem}.pdf", dpi=a.dpi)
    fig.savefig(f"{stem}.jpg", dpi=a.dpi, pil_kwargs={"quality": 88})
    plt.close(fig)
    jpeg_images(f"{stem}.pdf", a.jpeg_quality)
    print(f"[OK] -> {stem}.pdf , {stem}.jpg  ({W}x{H}, "
          f"plants near/mid/far = {counts['near']}/{counts['mid']}/{counts['far']})")


if __name__ == "__main__":
    main()
