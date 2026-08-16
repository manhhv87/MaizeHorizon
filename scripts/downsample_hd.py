#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ha mau khung 8K Hai Phong xuong 1920x1080 de tach bien do phan giai cam bien
ra khoi bien dia diem.

Doi chung cheo Dan Phuong <-> Hai Phong doi mot luc bon thu: dia diem, cam bien,
ong kinh, thoi diem. Khong tach duoc thu nao. Ha mau chinh khung 8K giu nguyen
canh, cay, anh sang va nhan, chi bo di do phan giai -- nen chenh lech con lai la
cua rieng cam bien.

Day KHONG phai suy giam tong hop dung thay cho anh xa that (thu ma bai dang phe
phan o phan distillation): anh 8K la that, ha mau chi dong vai ablation. Phai
khai ro nhu vay trong bai.

Nhan YOLO chuan hoa theo ti le anh nen giu nguyen, khong can sua.

    python scripts/downsample_hd.py --src <thu muc co images/ va labels/> --dst <dich>
"""

import argparse
import glob
import os
import shutil

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="thu muc chua images/ va labels/")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.dst, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.dst, "labels"), exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(a.src, "images", "*.jpg"))):
        b = os.path.basename(p)
        with Image.open(p) as im:
            im.convert("RGB").resize((a.width, a.height), Image.LANCZOS).save(
                os.path.join(a.dst, "images", b), quality=a.quality)
        lab = os.path.join(a.src, "labels", b.rsplit(".", 1)[0] + ".txt")
        if os.path.exists(lab):
            shutil.copy(lab, os.path.join(a.dst, "labels"))
        n += 1
    print(f"{n} anh -> {a.width}x{a.height} tai {a.dst}")


if __name__ == "__main__":
    main()
