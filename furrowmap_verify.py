#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thumbnail grid for human verification of the ledger.

Each ledger plant becomes one cell, cropped from the frame where it is largest and clearest, with
the target box outlined and its id shown. Cells are ordered so fragments of the same plant land
next to each other. The reviewer then marks: two identical cells = one plant counted twice, one
cell containing two plants = a merge, a cell with no plant = a false positive.

  python furrowmap_verify.py --ledger results/counting/ledger_IMG_3916_near.json \
      --frames-dir data/images/IMG_3916_test --out _verify/IMG_3916
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True, help="tien to file, vd _verify/IMG_3916 -> _verify/IMG_3916_page01.jpg")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=12, help="rows per page")
    ap.add_argument("--thumb", type=int, default=170)
    ap.add_argument("--pad", type=float, default=0.6, help="context around the box, as a fraction of its size, to show neighbours")
    a = ap.parse_args()
    import cv2

    led = json.load(open(a.ledger, encoding="utf-8"))
    plants = led["plants"]
    if not any(p.get("boxes") for p in plants):
        raise SystemExit("[X] ledger has no 'boxes' field; this is an old ledger.\n"
                         "    Chay LAI furrowmap_ledger.py (ban moi) de tao lai .json roi verify.")
    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")) +
                    glob.glob(os.path.join(a.frames_dir, "*.png")) +
                    glob.glob(os.path.join(a.frames_dir, "*.jpeg")))
    if not frames:
        raise SystemExit(f"[X] no frames found in {a.frames_dir}")

    # per plant: pick the tallest box, i.e. the nearest and clearest view
    cells = []
    for p in plants:
        if not p.get("boxes"):
            continue
        best = max(p["boxes"], key=lambda bx: bx[4] - bx[2])       # [fi,x1,y1,x2,y2], height=y2-y1
        fi, x1, y1, x2, y2 = best
        cells.append({"id": p["id"], "fi": fi, "box": [x1, y1, x2, y2], "cx": (x1 + x2) / 2})
    # order by (frame, x) so fragments of one plant sit together
    cells.sort(key=lambda c: (c["fi"], c["cx"]))

    img_cache = {}

    def get_img(fi):
        if fi not in img_cache:
            img_cache[fi] = cv2.imread(frames[fi]) if 0 <= fi < len(frames) else None
        return img_cache[fi]

    T = a.thumb
    per_page = a.cols * a.rows
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n_pages = (len(cells) + per_page - 1) // per_page
    n_drawn = 0
    for pg in range(n_pages):
        canvas = np.full((a.rows * T, a.cols * T, 3), 40, np.uint8)
        for k in range(per_page):
            gi = pg * per_page + k
            if gi >= len(cells):
                break
            c = cells[gi]; im = get_img(c["fi"])
            r, cc = divmod(k, a.cols); y0, x0 = r * T, cc * T
            if im is None:
                continue
            H, W = im.shape[:2]; x1, y1, x2, y2 = c["box"]
            bw, bh = x2 - x1, y2 - y1; pad = a.pad * max(bw, bh, 12)
            cx1 = max(0, int(x1 - pad)); cy1 = max(0, int(y1 - pad))
            cx2 = min(W, int(x2 + pad)); cy2 = min(H, int(y2 + pad))
            crop = im[cy1:cy2, cx1:cx2].copy()
            if crop.size == 0:
                continue
            # outline the target box in crop coordinates
            cv2.rectangle(crop, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (0, 255, 0), 2)
            # letterbox ve vuong T
            ch, cw = crop.shape[:2]; s = min((T - 4) / cw, (T - 4) / ch)
            nw, nh = max(1, int(cw * s)), max(1, int(ch * s))
            th = cv2.resize(crop, (nw, nh))
            cell = np.full((T, T, 3), 40, np.uint8)
            oy, ox = (T - nh) // 2, (T - nw) // 2
            cell[oy:oy + nh, ox:ox + nw] = th
            # so ID + POT
            cv2.rectangle(cell, (0, 0), (T, 20), (0, 0, 0), -1)
            cv2.putText(cell, f'#{c["id"]} f{c["fi"]}', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            canvas[y0:y0 + T, x0:x0 + T] = cell; n_drawn += 1
            cv2.rectangle(canvas, (x0, y0), (x0 + T - 1, y0 + T - 1), (90, 90, 90), 1)
        out = f"{a.out}_page{pg + 1:02d}.jpg"
        cv2.imwrite(out, canvas)
    absdir = os.path.abspath(os.path.dirname(a.out) or ".")
    print(f"[OK] {len(cells)} plants -> {n_pages} pages | {n_drawn} cells drawn")
    print(f"    FILE LUOI o: {absdir}/  ->  {os.path.basename(a.out)}_page01.jpg ...")
    if n_drawn < len(cells) * 0.5:
        print(f"    [!] many empty cells; --frames-dir may not match the ledger frame index")
    print("    review: two identical cells = one plant split (subtract); one cell with two plants = a merge (add);")
    print("    a cell with no plant = false positive (subtract). The result is the true plant count.")


if __name__ == "__main__":
    main()
