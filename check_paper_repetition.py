#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tim cac doan LAP Y trong ban thao, va cac so TU MAU THUAN.

  python check_paper_repetition.py                      # ban EN
  python check_paper_repetition.py --dir paper/vi/sections --lang vi
  python check_paper_repetition.py --strict             # exit 1 neu con phat hien

Hai loi nay khong bo kiem nao dang co bat duoc, va ca hai deu sinh ra tu cung mot
thoi quen: sua bai bang cach THEM doan moi thay vi sua doan cu.

  lap y        : hai cau o hai muc khac nhau noi cung mot dieu. Doc rieng thi cau nao
                 cung dung, doc lien thi thanh nhai lai. Do bang do trung n-gram sau
                 khi bo het lenh LaTeX va so.
  tu mau thuan : cung mot dai luong duoc bao cao bang hai gia tri khac nhau o hai cho
                 (vd h50@640 vua la 60,9 vua la 59,9). check_paper_numbers.py khong
                 thay, vi no doi chieu ban thao voi CSV chu khong doi chieu ban thao
                 voi CHINH NO.

Nguong mac dinh 0,35: du thap de bat cau viet lai bang tu khac ma van cung y, va
van tren muc trung ngau nhien cua hai cau cung chu de. Ha them se keo theo nhieu cap
chi trung thuat ngu.
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

# Cac lenh LaTeX phai bo TRUOC khi so, neu khong hai cau cung dat day \ref va \,px
# se trong giong nhau mot cach gia tao.
STRIP = [
    (re.compile(r"\\(?:label|ref|eqref|parencite|cite|citep|textcite)\{[^}]*\}"), " "),
    (re.compile(r"\\(?:emph|textbf|textit|text)\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\begin\{[^}]*\}|\\end\{[^}]*\}"), " "),
    (re.compile(r"\$[^$]*\$"), " NUM "),
    (re.compile(r"\\[a-zA-Z]+"), " "),
    (re.compile(r"[{}~\\%&]"), " "),
    (re.compile(r"\s+"), " "),
]


def clean(s):
    for pat, rep in STRIP:
        s = pat.sub(rep, s)
    return s.strip()


def sentences(path):
    """Tra ve (so_dong, cau) cho moi cau ngoai bang/hinh.

    Phai gom DOAN roi moi tach cau. Tach theo tung dong thi cau nao vat qua xuong
    dong se bi cat thanh manh ngan hon nguong va bien mat khoi phep so -- luc do
    ket qua chi phan anh cach ngat dong cua file chu khong phan anh noi dung, va
    hai ban dich cua cung mot doan se cho hai so khac han nhau.
    """
    out, depth, buf, start = [], 0, [], 1
    def flush():
        if not buf:
            return
        for s in re.split(r"(?<=[.!?])\s+", clean(" ".join(buf))):
            s = s.strip()
            if len(s.split()) >= 12:        # cau qua ngan thi trung la ngau nhien
                out.append((start, s))
        buf.clear()

    for i, raw in enumerate(open(path, encoding="utf-8"), 1):
        if re.search(r"\\begin\{(table|figure|tabular)", raw):
            flush(); depth += 1
        if re.search(r"\\end\{(table|figure|tabular)", raw):
            depth = max(0, depth - 1); buf.clear(); continue
        if depth or raw.lstrip().startswith("%"):
            continue
        if not raw.strip():                  # dong trong = het doan
            flush(); continue
        if not buf:
            start = i
        buf.append(raw.strip())
    flush()
    return out


def shingles(s, n=4):
    w = [x.lower() for x in re.findall(r"\w+", s) if x.upper() != "NUM"]
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="paper/sections")
    ap.add_argument("--lang", choices=("en", "vi"), default="en")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "*.tex"))
                   if not os.path.basename(f).startswith("_")
                   and os.path.basename(f) not in
                   ("preamble.tex", "nomenclature.tex", "coverpage.tex",
                    "title_authors.tex", "title_noauthors.tex", "declarations.tex"))
    items = []
    for f in files:
        for ln, s in sentences(f):
            items.append((os.path.basename(f)[:-4], ln, s, shingles(s)))
    print(f"=== {len(items)} cau tu {len(files)} file ===\n")

    # ---- lap y
    dup = []
    for (fa, la, sa, ga), (fb, lb, sb, gb) in combinations(items, 2):
        if not ga or not gb:
            continue
        j = len(ga & gb) / len(ga | gb)
        if j >= a.thr:
            dup.append((j, fa, la, sa, fb, lb, sb))
    dup.sort(reverse=True, key=lambda x: x[0])
    print(f"--- lap y ({len(dup)}) ---")
    for j, fa, la, sa, fb, lb, sb in dup[:14]:
        print(f"  [{j:.2f}] {fa}:{la}  <->  {fb}:{lb}")
        print(f"        A: {sa[:150]}")
        print(f"        B: {sb[:150]}")

    # ---- tu mau thuan: cung mot cum mo ta + so khac nhau
    dec = r"\d+[.,]\d+" if a.lang == "en" else r"\d+[,]\d+"
    QTY = {
        # `\$?` la bat buoc: trong .tex so nam trong $...$ nen sau chu so con mot
        # dau $ truoc `\,px`. Thieu no thi bo kiem im lang khong bat duoc gi.
        "h50 @640":      re.compile(r"\$640\$[^.]{0,60}?(" + dec + r")\$?\s*\\?,?\s*px"),
        "tam 8K @2560":  re.compile(r"\$2560\$[^.]{0,90}?(" + dec + r")\$?\s*\\?,?\s*m\b"),
    }
    text = "\n".join(open(f, encoding="utf-8").read() for f in files)
    clash = []
    for name, pat in QTY.items():
        vals = sorted({m.group(1) for m in pat.finditer(text)})
        if len(vals) > 1:
            clash.append((name, vals))
    print(f"\n--- tu mau thuan ({len(clash)}) ---")
    for name, vals in clash:
        print(f"  {name}: ban thao dung {', '.join(vals)}")

    n = len(dup) + len(clash)
    print(f"\n  {len(dup)} cap lap y | {len(clash)} dai luong tu mau thuan")
    if a.strict and n:
        sys.exit(1)


if __name__ == "__main__":
    main()
