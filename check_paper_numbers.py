#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Doi chieu tung con so trong ban thao voi CSV sinh ra no.

  python check_paper_numbers.py                 # ban EN
  python check_paper_numbers.py --dir paper/vi --lang vi
  python check_paper_numbers.py --strict        # exit 1 neu co FAIL
  python check_paper_numbers.py --orphans       # them: so trong bai khong thay o CSV nao

Vi sao can. `check_traceability.py` chi bao dam moi CSV truy duoc ve mot lenh; no
khong biet con so trong bai co khop CSV hay khong. Sau mot dot chay lai (vd doi tu
3 seed sang 5 seed) thi CSV doi con van ban thi khong, va khong gi bao loi ca --
PDF van dung, `??` van bang 0. Day la cho de mat dong bo nhat.

Hai che do:

  REGISTRY  moi muc noi mot o CSV cu the toi mot chuoi phai xuat hien trong .tex.
            Chinh xac, nhung phai khai bao tay. Dung cho cac con so dau de.
  ORPHANS   quet moi so trong .tex roi tim xem no co xuat hien o CSV nao khong.
            Bat duoc so cu sot lai, nhung nhieu bao dong gia (nam thang, so trang,
            tham so...), nen mac dinh tat.

Ban tieng Viet dung dau phay thap phan, nen so duoc doi dang truoc khi tim.
"""
import argparse
import csv
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def load(path):
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p, encoding="utf-8")))


def pick(rows, **where):
    """Dong dau tien khop moi dieu kien."""
    if rows is None:
        return None
    for r in rows:
        if all(str(r.get(k, "")).strip() == str(v) for k, v in where.items()):
            return r
    return None


def fmt(v, nd, lang):
    s = f"{float(v):.{nd}f}"
    return s.replace(".", "{,}") if lang == "vi" else s


def contains(text, needle):
    """Tim `needle` nhung KHONG cho no nam long trong mot so dai hon.

    Bay da dinh: so sanh chuoi con lam "0.93" khop voi "0.931" trong ban thao,
    nen mot bang con nguyen so cu van bao dat. Chan bang cach doi hai dau khong
    duoc la chu so (hoac dau thap phan tiep theo).
    """
    esc = re.escape(needle)
    pat = r"(?<![\d.,])" + esc + r"(?![\d]|\{,\}|\.\d)"
    return re.search(pat, text) is not None


# --- REGISTRY -----------------------------------------------------------------
# (nhan, duong dan CSV, dieu kien chon dong, cot, nd[, ctx])
#   nd  : so chu so thap phan; co the la tuple neu ban thao lam tron manh hon CSV
#   ctx : mau chuoi de tim, "{v}" duoc thay bang gia tri. Bat buoc khi nd=0, vi
#         mot so nguyen tran trui nhu "52" xuat hien khap noi va khong kiem duoc gi.
CHECKS = [
    # --- h50 va tam lam viec tren bo 8K (Bang tab:minlaw, Tom tat, Ket luan)
    ("8K h50 @1280", "results/crosssite/hd_h50_n5.csv", dict(imgsz="1280"), "h50_native_px", 1),
    ("8K h50 @1920", "results/crosssite/hd_h50_n5.csv", dict(imgsz="1920"), "h50_native_px", 1),
    ("8K h50 @2560", "results/crosssite/hd_h50_n5.csv", dict(imgsz="2560"), "h50_native_px", 1),
    ("8K h50 @3840", "results/crosssite/hd_h50_n5.csv", dict(imgsz="3840"), "h50_native_px", 1),

    # --- recall theo chieu cao goc tren 8K (than Bang tab:minlaw)
    ("8K recall 48-64 @1280", "results/crosssite/hd_sweep_phys.csv",
     dict(imgsz="1280", phys_lo_px="48"), "recall_mean", 3),
    ("8K recall 48-64 @3840", "results/crosssite/hd_sweep_phys.csv",
     dict(imgsz="3840", phys_lo_px="48"), "recall_mean", 3),
    ("8K recall 192+ @1280", "results/crosssite/hd_sweep_phys.csv",
     dict(imgsz="1280", phys_lo_px="192"), "recall_mean", 3),

    # --- precision tang xa tren 8K (Bang tab:minlaw, Tom tat)
    ("8K prec far @1280", "results/crosssite/hd_sweep_prec.csv",
     dict(imgsz="1280", tier="far"), "prec_mean", 3),
    ("8K prec far @2560", "results/crosssite/hd_sweep_prec.csv",
     dict(imgsz="2560", tier="far"), "prec_mean", 3),

    # --- h50 tren bo 1080p (khang dinh dau de: 52 px)
    # Bai lam tron 51.9 -> 52 co y: h50 noi suy giua diem giua cac bin rong 16 px,
    # nen chenh ~1 px khong co nghia. Kiem ca hai cach lam tron, kem ngu canh.
    ("1080p h50 @1280", "results/scaling/scaling_range.csv",
     dict(imgsz="1280"), "h50_phys_px", (1, 0), "${v}$\\,px"),

    # --- AP theo tang, IoU 0.3 (Bang tab:detection). Bang in 3 chu so thap phan;
    # kiem o 2 chu so tung cho khop nham vi "0.93" nam long trong "0.931".
    ("AP stock near", "results/detection/testset_ap03.csv",
     dict(model="stock", stratum="near"), "ap_mean", 3),
    ("AP stock far", "results/detection/testset_ap03.csv",
     dict(model="stock", stratum="far"), "ap_mean", 3),
    ("AP nearfar mid", "results/detection/testset_ap03.csv",
     dict(model="nearfar", stratum="mid"), "ap_mean", 3),
    ("AP distill far", "results/detection/testset_ap03.csv",
     dict(model="distill", stratum="far"), "ap_mean", 3),
    # --- tran recall theo tang (Bang tab:cliff)
    ("cliff stock far ceiling", "results/detection/testset_ceiling.csv",
     dict(model="stock", stratum="far"), "recall_mean", 3),
    ("cliff distill far @0.25", "results/detection/testset_iou03.csv",
     dict(model="distill", stratum="far"), "recall_mean", 3),
    # --- da kien truc (Bang tab:multiarch)
    ("multiarch rtdetr far", "results/baselines/rebuttal_multiarch.csv",
     dict(arch="rtdetr-l", stratum="far"), "recall_mean", 3),
    ("multiarch yolo11s ceiling", "results/baselines/rebuttal_multiarch_far_ceiling.csv",
     dict(arch="yolo11s"), "far_recall_ceiling", 3),

    # --- phep dao dau cat o (Muc res-tiling)
    # Tro vao ban _fpfilt: eval_ap.py nay loc kich thuoc cho false positive dung
    # chuan COCO, nen moi AP theo tang doi so. Ban khong loc van con tren dia de
    # doi chieu, nhung bai KHONG trich dan no nua.
    ("tiling 8K full AP", "results/crosssite/hd_sahi_n5_fpfilt.csv",
     dict(tier="far"), "full_ap_mean", 3),
    ("tiling 8K sahi AP", "results/crosssite/hd_sahi_n5_fpfilt.csv",
     dict(tier="far"), "sahi_ap_mean", 3),
    ("tiling 1080p full AP", "results/baselines/sahi_ap_n5_fpfilt.csv",
     dict(tier="far"), "full_ap_mean", 3),
    ("tiling 1080p sahi AP", "results/baselines/sahi_ap_n5_fpfilt.csv",
     dict(tier="far"), "sahi_ap_mean", 3),
    # Bo o "near sau cat o": bai khong con trich dan no sau khi cat o chuyen thanh
    # ket qua am (huong A). Giu hai o far vi bai van bao cao ca hai.
    ("tiling E5 ha mau sahi", "results/crosssite/E5_sahi_8k_ds1080_n5_fpfilt.csv",
     dict(tier="far"), "sahi_ap_mean", 3),

    # --- thong ke cua phep dao dau. Dang ky truoc day chi co gia tri AP, nen khi
    # phep kiem E5 doi tu Welch sang ghep cap (t 3,85 -> 4,80; p 0,012 -> 0,0086)
    # thi khong tang nao bat duoc bai dang mang so cu. Nay co ca t va p.
    ("tiling tuong tac 1080p-8K", "results/rebuttal/tiling_reversal_stats_fpfilt.csv",
     dict(corpus="interaction"), "diff", 3),
    ("tiling tuong tac t", "results/rebuttal/tiling_reversal_stats_fpfilt.csv",
     dict(corpus="interaction"), "t", (2, 1)),
    ("E5 tuong tac diff", "results/rebuttal/E5_control_stats_fpfilt.csv",
     dict(corpus="interaction"), "diff", 3),
    ("E5 tuong tac t", "results/rebuttal/E5_control_stats_fpfilt.csv",
     dict(corpus="interaction"), "t", (2, 1)),

    # --- do nhay cua h50 theo nguong bin thua (muc sec:res-minn). Nguong nay tung
    # KHONG duoc khai bao trong Methods; nay bai bao cao ca uoc luong don dieu.
    ("h50 iso @1280", "results/crosssite/hd_h50_isotonic.csv",
     dict(imgsz="1280"), "h50_isotonic", 1),
    ("h50 iso @2560", "results/crosssite/hd_h50_isotonic.csv",
     dict(imgsz="2560"), "h50_isotonic", 1),
    ("h50 iso @3840", "results/crosssite/hd_h50_isotonic.csv",
     dict(imgsz="3840"), "h50_isotonic", 1),

    # --- diem gay cua quy luat min (M8, muc sec:res-native)
    ("h50 stock 1920 vs 960", "results/rebuttal/M8_h50_crossover.csv",
     dict(conf="0.25", tag_a="stock", imgsz_a="1920", imgsz_b="960"), "delta", 2),
    ("h50 nearfar 1920 vs 960", "results/rebuttal/M8_h50_crossover.csv",
     dict(conf="0.25", tag_a="nearfar", imgsz_a="1920", imgsz_b="960"), "delta", 2),
    ("h50 stock tren goc 2560", "results/rebuttal/M8_h50_crossover.csv",
     dict(conf="0.001", tag_a="stock", imgsz_a="2560"), "delta", 2),
    ("h50 nwd tren goc 2560", "results/rebuttal/M8_h50_crossover.csv",
     dict(conf="0.001", tag_a="nwd", imgsz_a="2560"), "delta", 2),
    ("h50 nwd so voi stock", "results/rebuttal/M8_h50_crossover.csv",
     dict(conf="0.25", tag_a="nwd", tag_b="stock"), "delta", 2),

    # --- NWD la biet le duong, khong con la ket qua am
    ("AP nwd far", "results/detection/testset_ap03.csv",
     dict(model="nwd", stratum="far"), "ap_mean", 3),
    ("AP nwd mid", "results/detection/testset_ap03.csv",
     dict(model="nwd", stratum="mid"), "ap_mean", 3),

    # --- Chuyen giao zero-shot sang bo 8K. Ba o nay tung lech giua hai ban dich:
    # ban EN dung so 5 seed con ban VI con giu so 3 seed (0,916/0,678/0,183), va
    # khong bo kiem nao thay, vi chung khong nam trong dang ky. Nay co.
    ("8K zero-shot near", "results/crosssite/namsach_ceiling.csv",
     dict(model="stock", stratum="near"), "recall_mean", 3),
    ("8K zero-shot mid", "results/crosssite/namsach_ceiling.csv",
     dict(model="stock", stratum="mid"), "recall_mean", 3),
    ("8K zero-shot far", "results/crosssite/namsach_ceiling.csv",
     dict(model="stock", stratum="far"), "recall_mean", 3),

    # --- h50 cua bo 8K o hai dau lot quet (VI tung ghi 110,9 -> 60,5)
    ("8K h50 @1280 (1 chu so)", "results/crosssite/hd_h50_n5.csv",
     dict(imgsz="1280"), "h50_native_px", 1),
    ("8K h50 @3840 (1 chu so)", "results/crosssite/hd_h50_n5.csv",
     dict(imgsz="3840"), "h50_native_px", 1),

    # --- h50 theo POT giua hai dia diem (VI tung ghi 20,4 so voi 18,7)
    ("h50 POT danphuong", "results/crosssite/hd_h50_pot.csv",
     dict(label="danphuong"), "h50_pot_px", 1),
    ("h50 POT haiphong ha mau", "results/crosssite/hd_h50_pot.csv",
     dict(label="haiphong_ds1080"), "h50_pot_px", 1),

    # --- ngan sach giam sat (Bang tab:supervision)
    ("supervision scale025 far", "results/rebuttal/M4_data_scaling.csv",
     dict(arm="scale025"), "recall_far", 3),
    ("supervision nearfar far", "results/rebuttal/M4_data_scaling.csv",
     dict(arm="nearfar"), "recall_far", 3),
    ("supervision farx20 far", "results/rebuttal/M4b_oversample_far.csv",
     dict(arm="farx20"), "recall_far", 3),

    # --- so luong nhan (Phuong phap)
    ("so cay rieng biet", "results/dataset/distinct_plants.csv",
     dict(tier="all"), "n_plant", 0),
    ("so box tang xa", "results/dataset/distinct_plants.csv",
     dict(tier="far"), "n_box", 0),
]


def tex_text(d):
    """Noi moi .tex trong thu muc sections/ thanh mot chuoi."""
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".tex"):
            out.append(open(os.path.join(d, f), encoding="utf-8").read())
    return "\n".join(out)


def scan_orphans(text, lang):
    """So trong .tex khong xuat hien o bat ky CSV nao. Nhieu bao dong gia."""
    pool = set()
    for root, _, files in os.walk(os.path.join(REPO, "results")):
        if "n3_seed3_snapshot" in root:
            continue
        for f in files:
            if not f.endswith(".csv"):
                continue
            for line in open(os.path.join(root, f), encoding="utf-8", errors="replace"):
                for m in re.finditer(r"\d+\.\d+", line):
                    v = m.group(0)
                    pool.add(v)
                    for nd in (1, 2, 3, 4):          # ban thao co the lam tron
                        pool.add(f"{float(v):.{nd}f}")
    sep = r"\{,\}" if lang == "vi" else r"\."
    pat = re.compile(r"\$(\d+" + sep + r"\d+)\$")
    seen, orph = set(), []
    for m in pat.finditer(text):
        v = m.group(1).replace("{,}", ".")
        if v in seen:
            continue
        seen.add(v)
        if v not in pool:
            orph.append(v)
    return orph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="paper/sections")
    ap.add_argument("--lang", choices=("en", "vi"), default="en")
    ap.add_argument("--orphans", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    d = os.path.join(REPO, a.dir)
    if not os.path.isdir(d):
        raise SystemExit(f"khong thay {d}")
    text = tex_text(d)

    ok = fail = miss = 0
    print(f"=== doi chieu {a.dir} voi results/ ===")
    for entry in CHECKS:
        label, path, where, col, nd = entry[:5]
        ctx = entry[5] if len(entry) > 5 else None
        nds = nd if isinstance(nd, tuple) else (nd,)
        rows = load(path)
        if rows is None:
            print(f"  [?] {label:26s} khong co {path}")
            miss += 1
            continue
        r = pick(rows, **where)
        if r is None or col not in r:
            print(f"  [?] {label:26s} khong thay dong {where} cot {col}")
            miss += 1
            continue
        cands = [fmt(r[col], k, a.lang) for k in nds]
        hits = [c for c in cands
                if (ctx.replace("{v}", c) in text if ctx else contains(text, c))]
        if hits:
            ok += 1
        else:
            fail += 1
            shown = " hoac ".join(ctx.replace("{v}", c) if ctx else c for c in cands)
            print(f"  [X] {label:26s} CSV co {shown}, ban thao KHONG co")
            print(f"      {path} {where} .{col}")
    print(f"\n  {ok} khop | {fail} lech | {miss} khong tra duoc")

    if a.orphans:
        orph = scan_orphans(text, a.lang)
        print(f"\n=== so trong ban thao khong thay o CSV nao ({len(orph)}) ===")
        print("  (nhieu bao dong gia: nam, so trang, tham so, gia tri suy ra)")
        for v in orph[:60]:
            print(f"    {v}")
        if len(orph) > 60:
            print(f"    ... con {len(orph)-60} nua")

    if a.strict and (fail or miss):
        sys.exit(1)


if __name__ == "__main__":
    main()
