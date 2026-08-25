#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Doi chieu cac bang trong PDF DA DUNG voi CSV sinh ra chung.

  python check_pdf_tables.py                       # ban EN
  python check_pdf_tables.py --pdf paper/vi/main_vi.pdf --lang vi
  python check_pdf_tables.py --strict              # exit 1 neu lech

Vi sao doc PDF chu khong doc .tex. `make_tables.py --check` da so .tex voi CSV, nhung
no so voi chinh chuoi ma no sinh ra -- neu generator sai thi ca hai deu sai cung mot
kieu va khong ai bat duoc. Doc PDF la duong doc lap: no la thu nguoi phan bien thuc
su cam tren tay, va no di qua ca LaTeX (co the nuot dong, gop o, cat bang).

Phep thu dung vi tri, khong dung "co mat dau do". Mot gia tri co the trung ngau nhien
o cho khac trong bai; chi khi no nam DUNG o thu k cua dong bat dau bang nhan cua no
thi moi ket luan la bang dung.

Bay khi doc: pdftotext tra ve ca gia tri lan do lech chuan tren cung mot dong
("0.928 0.013 0.458 ..."), nen phai lay cach quang chu khong lay lien tiep.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def load(p, *keys):
    f = os.path.join(REPO, p)
    if not os.path.exists(f):
        return None
    return {tuple(x[k] for k in keys): x for x in csv.DictReader(open(f, encoding="utf-8"))}


def fmt(v, lang, nd=3):
    s = f"{float(v):.{nd}f}"
    return s.replace(".", ",") if lang == "vi" else s


def pdf_lines(pdf):
    out = subprocess.run(["pdftotext", "-layout", pdf, "-"],
                         capture_output=True, text=True).stdout
    return [L.strip() for L in out.split("\n")]


def region(lines, anchor):
    """Chi so dong dau tien chua `anchor`; dung de khoanh vung mot bang.

    Nhieu bang dung chung nhan dong ("Stock", "48-64"), nen tim tu dau file se bat
    nham bang khac va bao lech gia. Luon neo vao mot cum chi co trong caption cua
    dung bang can kiem.
    """
    for i, s in enumerate(lines):
        if anchor in s:
            return i
    return None


def row_numbers(lines, label, need, lang, after=0, exact=None):
    """So thap phan tren dong dau tien SAU `after` bat dau bang `label`.

    `exact` buoc dong phai co dung ngan ay so, de phan biet hai bang co cung nhan
    dong nhung khac so cot.
    """
    pat = r"\d,\d{3}" if lang == "vi" else r"\d\.\d{3}"
    for s in lines[after:]:
        if s.startswith(label):
            v = re.findall(pat, s)
            if exact is not None and len(v) != exact:
                continue
            if len(v) >= need:
                return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="paper/main.pdf")
    ap.add_argument("--lang", choices=("en", "vi"), default="en")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    pdf = os.path.join(REPO, a.pdf)
    if not os.path.exists(pdf):
        raise SystemExit(f"khong thay {pdf}")
    lines = pdf_lines(pdf)
    # Neo tung bang vao mot cum chi co trong caption cua no. Thieu buoc nay thi mot
    # nhan dong dung chung ("Stock", "48-64") se bat nham bang khac.
    # Cum neo phai lay tu chinh ban DA DUNG (`pdftotext -layout`), khong phai tu
    # nguon .tex: LuaLaTeX ngat dong caption va tach dong tieu de thanh nhieu cot,
    # nen mot cum dung trong .tex co the khong bao gio xuat hien tron ven tren mot
    # dong cua PDF. Neo hut thi region() lui ve 0 va phep thu bat nham bang khac.
    ANCHORS = {
        "ap":     ("Per-tier average precision", "AP phát hiện theo từng tầng"),
        "arch":   ("Cross-architecture control", "Đối chứng cross-architecture"),
        "cliff":  ("Far recall @conf", "Recall tầng xa trên bộ Đan Phượng"),
        # "(px)" la phan bat buoc: bang train-matched cung mo dau bang "Chieu cao
        # goc", nen neo thieu "(px)" se bat vao bang do va bao lech gia.
        "8k":     ("Native height (px)", "Chiều cao gốc (px)"),
    }
    # Neo hut phai BAO, khong duoc im lang lui ve 0: lui ve 0 nghia la tim tu dau
    # file, bat nham mot bang khac co cung nhan dong, roi bao "lech" voi nhung con
    # so khong lien quan. Da dinh mot lan voi bang 8K ban VI.
    idxs, lost = {}, []
    for k, (en, vi) in ANCHORS.items():
        idxs[k] = region(lines, en if a.lang == "en" else vi)
        if idxs[k] is None:
            lost.append(f"{k} (tim '{en if a.lang == 'en' else vi}')")
            idxs[k] = 0
    if lost:
        print(f"  [!] khong tim thay neo cho: {', '.join(lost)}")
        print("      phep thu duoi day se tim tu dau file va co the bat nham bang.\n")
    r_ap, r_arch, r_cliff, r_8k = idxs["ap"], idxs["arch"], idxs["cliff"], idxs["8k"]
    F = lambda v, nd=3: fmt(v, a.lang, nd)
    bad = ok = skip = 0

    def report(name, want, got):
        nonlocal bad, ok
        if got == want:
            ok += 1
            print(f"  [OK] {name}")
        else:
            bad += 1
            print(f"  [X]  {name}\n       PDF={got}\n       CSV={want}")

    # ---- Bang AP theo tang: gia tri va SD xen ke -> lay cach quang
    a3 = load("results/detection/testset_ap03.csv", "model", "stratum")
    a5 = load("results/detection/testset_ap05.csv", "model", "stratum")
    if a3 and a5:
        for m, lab in (("stock", "Stock"), ("nearfar", "+Mint"),
                       ("distill", "+Distill"), ("distill_shuffle", "+Distill (shuffle)")):
            want = [F(a3[(m, t)]["ap_mean"]) for t in ("near", "mid", "far")] + \
                   [F(a5[(m, t)]["ap_mean"]) for t in ("near", "mid", "far")]
            v = row_numbers(lines, lab, 12, a.lang, after=r_ap)
            report(f"AP theo tang: {lab}", want, v[0::2][:6] if v else None)
    else:
        skip += 1

    # ---- Bang da kien truc: 3 tang co SD, roi mot cot ceiling khong SD
    ma = load("results/baselines/rebuttal_multiarch.csv", "arch", "stratum")
    ce = load("results/baselines/rebuttal_multiarch_far_ceiling.csv", "arch")
    if ma and ce:
        for arch, lab in (("stock", "YOLOv8-s"), ("yolo11s", "YOLO11-s"),
                          ("yolov10s", "YOLOv10-s"), ("rtdetr-l", "RT-DETR-l")):
            want = [F(ma[(arch, t)]["recall_mean"]) for t in ("near", "mid", "far")] + \
                   [F(ce[(arch,)]["far_recall_ceiling"])]
            v = row_numbers(lines, lab, 7, a.lang, after=r_arch)
            report(f"da kien truc: {lab}", want, (v[0::2][:3] + [v[6]]) if v and len(v) >= 7 else None)
    else:
        skip += 1

    # ---- Bang tran recall: hai cot, khong co SD
    op = load("results/detection/testset_iou03.csv", "model", "stratum")
    cl = load("results/detection/testset_ceiling.csv", "model", "stratum")
    if op and cl:
        for m, lab in (("stock", "Stock"), ("nearfar", "+Mint")):
            want = [F(op[(m, "far")]["recall_mean"]), F(cl[(m, "far")]["recall_mean"])]
            hit = row_numbers(lines, lab, 2, a.lang, after=r_cliff, exact=2)
            report(f"tran recall: {lab}", want, hit)
    else:
        skip += 1

    # ---- Bang 8K theo chieu cao goc
    hd = load("results/crosssite/hd_sweep_phys.csv", "imgsz", "phys_lo_px")
    if hd:
        for lo, lab in (("48", "48–64"), ("64", "64–96"), ("96", "96–128")):
            want = [F(hd[(z, lo)]["recall_mean"]) for z in ("1280", "1920", "2560", "3840")]
            v = row_numbers(lines, lab, 8, a.lang, after=r_8k)
            report(f"8K, bin {lab}", want, v[0::2][:4] if v else None)
    else:
        skip += 1

    print(f"\n  {ok} dong khop | {bad} lech | {skip} nhom bo qua (thieu CSV)")
    if a.strict and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
