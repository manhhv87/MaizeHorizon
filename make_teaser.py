#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teaser figure: one forward-facing frame with boxes coloured by POT tier.

near (>= 32 px) green, mid (16-32) orange, far (< 16) red; ignore regions shaded. Two insets show
one near and one far plant at the same display size, so the scale difference is visible.

No GPU needed: one frame and its YOLO label file are enough.
"""
import argparse
import os

import numpy as np

from eval_testset import read_gt

IMGSZ = 1280
# BGR
COL = {"near": (60, 200, 60), "mid": (30, 140, 240), "far": (40, 40, 220)}
# Nhan theo ngon ngu, cho ban tieng Viet cua paper (paper/vi). Mac dinh 'en'
# nen anh cua ban English khong doi.
L10N = {
    "en": {"near": "near (>=32px)", "mid": "mid (16-32px)", "far": "far (<16px)",
           "near view": "near view", "far view": "far view"},
    "vi": {"near": "gần (≥ 32 px)", "mid": "giữa (16–32 px)", "far": "xa (< 16 px)",
           "near view": "khung nhìn gần", "far view": "khung nhìn xa"},
}


def put_text(im, txt, org, scale, color, thick, pil=False):
    """Ve chu len anh BGR.

    Font Hershey cua OpenCV chi co ASCII nen dau tieng Viet se ra o vuong; voi
    pil=True thi ve bang PIL voi DejaVu Sans (co du glyph). Quyet dinh theo
    NGON NGU chu khong theo noi dung chuoi: mot nhan tieng Viet co the tinh co
    khong co dau ("xa (< 16 px)"), va neu doi font theo tung chuoi thi cung mot
    chu giai se ra hai font khac nhau.
    """
    import cv2
    if not pil:
        cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        return im
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib.font_manager import findfont, FontProperties
    px = max(10, int(round(scale * 42)))          # xap xi chieu cao chu cua Hershey
    font = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), px)
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    # org cua cv2 la chan chu; PIL neo o dinh -> dung anchor "ls" (left baseline)
    d.text(org, txt, font=font, fill=(color[2], color[1], color[0]), anchor="ls",
           stroke_width=max(0, thick - 2), stroke_fill=(0, 0, 0))
    im[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return im


def tier_of(box, w, h):
    pot = (box[3] - box[1]) * IMGSZ / max(w, h)
    return "near" if pot >= 32 else ("mid" if pot >= 16 else "far"), pot


def crop_inset(im, box, pad_frac=0.6, size=200):
    import cv2
    H, W = im.shape[:2]
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    cx1 = int(max(0, x1 - px)); cy1 = int(max(0, y1 - py))
    cx2 = int(min(W, x2 + px)); cy2 = int(min(H, y2 + py))
    crop = im[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return None
    # draw boxes inside the crop
    bx1, by1 = int(x1 - cx1), int(y1 - cy1)
    bx2, by2 = int(x2 - cx1), int(y2 - cy1)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST), (bx1, by1, bx2, by2, crop.shape[1], crop.shape[0])


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="maizehorizon_example.jpg")
    ap.add_argument("--inset", type=int, default=220, help="canh inset (px)")
    ap.add_argument("--plant-class", type=int, default=0)
    ap.add_argument("--ignore-class", type=int, default=1)
    ap.add_argument("--lang", choices=("en", "vi"), default="en", help="ngon ngu nhan")
    a = ap.parse_args()
    T = L10N[a.lang]

    im = cv2.imread(a.image)
    if im is None:
        raise SystemExit(f"could not read image: {a.image}")
    H, W = im.shape[:2]
    plants, ignores = read_gt(a.label, W, H, a.plant_class, a.ignore_class)
    if not plants:
        raise SystemExit(f"no plants in label file: {a.label}")

    over = im.copy()
    # ignore mo xam
    for b in ignores:
        cv2.rectangle(over, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (150, 150, 150), -1)
    im = cv2.addWeighted(over, 0.25, im, 0.75, 0)
    # plant boxes coloured by tier
    counts = {"near": 0, "mid": 0, "far": 0}
    for b in plants:
        t, _ = tier_of(b, W, H)
        counts[t] += 1
        th = 3 if t == "near" else (2 if t == "mid" else 2)
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COL[t], th)

    # chon 1 near (cao nhat) + 1 far (thap nhat) lam inset
    by_h = sorted(plants, key=lambda b: b[3] - b[1])
    far_b = by_h[0]; near_b = by_h[-1]
    insets = []
    for b, key in [(near_b, "near view"), (far_b, "far view")]:
        r = crop_inset(im, b, size=a.inset)
        if r is not None:
            insets.append((r[0], r[1], key))

    # place insets top-right, stacked
    margin = 12
    y0 = margin
    for crop, (bx1, by1, bx2, by2, cw, ch), key in insets:
        s = a.inset
        # redraw the box inside the inset
        sx, sy = s / cw, s / ch
        cv2.rectangle(crop, (int(bx1 * sx), int(by1 * sy)), (int(bx2 * sx), int(by2 * sy)),
                      (0, 230, 230), 2)
        cv2.rectangle(crop, (0, 0), (s - 1, s - 1), (255, 255, 255), 2)
        put_text(crop, T[key], (6, 20), 0.6, (255, 255, 255), 2, pil=a.lang != "en")
        x0 = W - s - margin
        if y0 + s <= H:
            im[y0:y0 + s, x0:x0 + s] = crop
            # connector from inset to the original box
            b = near_b if key == "near view" else far_b
            cv2.line(im, (x0, y0 + s // 2), (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)),
                     (0, 230, 230), 1, cv2.LINE_AA)
            y0 += s + margin

    # tier legend, top-left
    yy = margin + 18
    for t in ("near", "mid", "far"):
        put_text(im, f"{T[t]}: {counts[t]}", (margin, yy), 0.62, COL[t], 2, pil=a.lang != "en")
        yy += 26

    cv2.imwrite(a.out, im)
    print(f"[OK] -> {a.out}  ({W}x{H}, plants near/mid/far = {counts['near']}/{counts['mid']}/{counts['far']})")


if __name__ == "__main__":
    main()
