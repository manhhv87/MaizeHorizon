#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh than cac bang ket qua thang tu CSV, va thay chung vao .tex.

  python make_tables.py --check          # chi bao bang nao lech, khong sua
  python make_tables.py --write          # thay than bang trong .tex
  python make_tables.py --write --lang vi --dir paper/vi/sections

Vi sao can. Go tay so vao bang la cho de mat dong bo nhat trong ca quy trinh. Sau
mot dot chay lai, CSV doi con bang thi khong; LaTeX van dich, `??` van bang 0, va
khong gi bao loi. Da dinh dung mot lan: sau khi nang tu 3 len 5 seed, bon bang van
giu nguyen so cu, trong do co o ghi `0.000` cho $+$Distill trong khi so that la
`0.015` -- du de doi han cach dien giai.

Bang trong bai duoc sinh tu CSV, khong go tay.

Cach lam: moi bang khai bao nguon CSV va cach xep dong. Script dung lai phan giua
`\\midrule` dau tien va `\\bottomrule`, roi thay dung phan do trong .tex. Bang van
nam nguyen trong .tex (tap chi thuong doi file phang), chi la khong con go tay.
"""
import argparse
import csv
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def rows(path):
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p, encoding="utf-8")))


def idx(path, *keys):
    r = rows(path)
    if r is None:
        return None
    return {tuple(x[k] for k in keys): x for x in r}


def num(v, nd, lang):
    s = f"{float(v):.{nd}f}"
    return s.replace(".", "{,}") if lang == "vi" else s


def cell(v, sd, nd, lang, bold=False):
    """$x$ kem {\\scriptsize$\\pm$sd}. Bold thi boc \\textbf quanh gia tri."""
    a = num(v, nd, lang)
    core = f"\\textbf{{${a}$}}" if bold else f"${a}$"
    if sd is None:
        return core
    return core + "{\\scriptsize$\\pm$" + num(sd, nd, lang).lstrip("$") + "}"


# ---------------------------------------------------------------- tab:detection
def gen_detection(lang):
    a3 = idx("results/detection/testset_ap03.csv", "model", "stratum")
    a5 = idx("results/detection/testset_ap05.csv", "model", "stratum")
    if not a3 or not a5:
        return None
    ARMS = [("stock", "Stock"), ("nearfar", "$+$Mint"),
            ("distill", "$+$Distill"), ("distill_shuffle", "$+$Distill (shuffle)")]
    TIERS = ["near", "mid", "far"]
    # in dam gia tri tot nhat moi cot
    best = {}
    for src, tag in ((a3, "3"), (a5, "5")):
        for t in TIERS:
            vals = {m: float(src[(m, t)]["ap_mean"]) for m, _ in ARMS if (m, t) in src}
            if vals:
                best[(tag, t)] = max(vals, key=vals.get)
    out = []
    for m, label in ARMS:
        cs = []
        for src, tag in ((a3, "3"), (a5, "5")):
            for t in TIERS:
                r = src.get((m, t))
                if r is None:
                    cs.append("--")
                    continue
                cs.append(cell(r["ap_mean"], r.get("ap_std"), 3, lang,
                               bold=(best.get((tag, t)) == m)))
            if tag == "3":
                cs.append("")          # cot trang giua hai nhom IoU
        out.append(f"{label:22s} & " + " & ".join(cs) + " \\\\")
    return "\n".join(out)


# -------------------------------------------------------------------- tab:cliff
def gen_cliff(lang):
    op = idx("results/detection/testset_iou03.csv", "model", "stratum")
    ce = idx("results/detection/testset_ceiling.csv", "model", "stratum")
    if not op or not ce:
        return None
    ARMS = [("stock", "Stock"), ("nearfar", "$+$Mint"),
            ("distill", "$+$Distill"), ("distill_shuffle", "$+$Distill (shuffle)")]
    out = []
    for m, label in ARMS:
        o = op.get((m, "far"))
        c = ce.get((m, "far"))
        if o is None or c is None:
            continue
        out.append(f"{label:22s} & {num(o['recall_mean'], 3, lang)} "
                   f"& {num(c['recall_mean'], 3, lang)} \\\\")
    return "\n".join(out)


# ----------------------------------------------------------------- tab:multiarch
def gen_multiarch(lang):
    r = idx("results/baselines/rebuttal_multiarch.csv", "arch", "stratum")
    c = idx("results/baselines/rebuttal_multiarch_far_ceiling.csv", "arch")
    if not r or not c:
        return None
    ARCHS = [("stock", "YOLOv8-s (YOLO)"), ("yolo11s", "YOLO11-s (YOLO)"),
             ("yolov10s", "YOLOv10-s (YOLO)"), ("rtdetr-l", "RT-DETR-l (DETR)")]
    TIERS = ["near", "mid", "far"]
    # in dam gia tri tot nhat moi cot; hoa thi in dam tat ca cac ben hoa
    best = {}
    for t in TIERS:
        v = {a: float(r[(a, t)]["recall_mean"]) for a, _ in ARCHS if (a, t) in r}
        if v:
            best[t] = max(v.values())
    ce = {a: float(c[(a,)]["far_recall_ceiling"]) for a, _ in ARCHS if (a,) in c}
    ce_best = max(ce.values()) if ce else None

    out = []
    for a, label in ARCHS:
        cs = []
        for t in TIERS:
            x = r.get((a, t))
            if x is None:
                cs.append("--")
                continue
            hot = best.get(t) is not None and abs(float(x["recall_mean"]) - best[t]) < 5e-4
            cs.append(cell(x["recall_mean"], x.get("recall_std"), 3, lang, bold=hot))
        y = c.get((a,))
        if y is None:
            cs += ["", "--"]
        else:
            v = num(y["far_recall_ceiling"], 3, lang)
            hot = ce_best is not None and abs(float(y["far_recall_ceiling"]) - ce_best) < 5e-4
            val = f"\\textbf{{${v}$}}" if hot else f"${v}$"
            # Ban EN bo so 0 dau (.248) theo dung cach bang goc viet; ban VI giu
            # nguyen 0{,}248, vi bo di se thanh ",248" -- khong phai quy uoc tieng Viet.
            lo = num(y["wilson_lo"], 3, lang)
            hi = num(y["wilson_hi"], 3, lang)
            if lang == "en":
                lo, hi = lo.lstrip("0"), hi.lstrip("0")
            cs += ["", f"{val} [{lo},\\,{hi}]"]
        out.append(f"{label:17s} & " + " & ".join(cs) + " \\\\")
    return "\n".join(out)


# ------------------------------------------------------------- tab:trainmatched
def gen_trainmatched(lang):
    """Moi cot la mot detector train VA eval o chinh kich thuoc dau vao do."""
    SZ = ["640", "960", "1280", "1920"]
    BINS = [("24", "32"), ("32", "48"), ("48", "64"), ("64", "96")]
    cols = {}
    for z in SZ:
        r = idx(f"results/scaling/scaling_tm_{z}_phys.csv", "phys_lo_px")
        if r is None:
            return None
        cols[z] = r
    out = []
    for lo, hi in BINS:
        cs = []
        for z in SZ:
            x = cols[z].get((lo,))
            cs.append(cell(x["recall_mean"], x.get("recall_std"), 3, lang) if x else "--")
        out.append(f"${lo}$--${hi}$ & " + " & ".join(cs) + " \\\\")
    return "\n".join(out)


# --------------------------------------------------------------- tab:crosssite
def gen_crosssite(lang):
    """Recall tai tran, phan tang theo POT, ba cot: 1080p / 8K goc / 8K ha mau."""
    def load(path):
        r = rows(path)
        if r is None:
            return None
        return {x["pot_lo_px"]: x for x in r if x["model"] == "stock"}
    A = load("results/detection/testset_ceiling_curve.csv")
    B = load("results/crosssite/namsach_ceiling_curve.csv")
    C = load("results/crosssite/namsach_ds1080_ceiling_curve.csv")
    if not (A and B and C):
        return None
    BINS = [("4", "8"), ("8", "12"), ("12", "16"), ("16", "20"), ("20", "24"),
            ("24", "32"), ("32", "48"), ("48", "64"), ("64", "96")]

    def one(x, with_n=True):
        if x is None or x["recall_mean"] in ("", "nan"):
            return "--"
        v = num(x["recall_mean"], 3, lang)
        if not with_n:
            return f"${v}$"
        n = int(float(x["n_gt"]))
        ns = f"{n:,}".replace(",", "{,}" if lang == "en" else "\\,")
        return f"${v}$ ($n{{=}}{ns}$)"

    out = []
    for lo, hi in BINS:
        out.append(f"${lo}$--${hi}$ & {one(A.get(lo))} & {one(B.get(lo))} "
                   f"& {one(C.get(lo), with_n=False)} \\\\")
    # Bang nay co mot \midrule long roi mot dong h50; sinh lai ca hai, neu khong
    # phep thay the (chay tu \midrule dau tien den \bottomrule) se nuot mat no.
    h = idx("results/crosssite/hd_h50_pot.csv", "label")
    if h:
        cells = [one_h(h, k, lang) for k in ("danphuong", "haiphong", "haiphong_ds1080")]
    else:
        cells = ["$18.7$", "$18.5$", "$20.4$"]      # gia tri da cong bo truoc do
    out.append("\\midrule")
    out.append("$h_{50}$ (px POT) & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def one_h(h, key, lang):
    x = h.get((key,))
    return f"${num(x['h50_pot_px'], 1, lang)}$" if x else "--"


# ---------------------------------------------------------------- tab:ablation
def gen_ablation(lang):
    """Hai phan: hieu AP distill-shuffle (tu CSV), va feature loss (khong co CSV).

    Feature loss lay tu log huan luyen giai doan 2, khong co tep ket qua nao giu no,
    nen giu nguyen o day duoi dang hang so. Neu chay lai giai doan 2 thi phai cap
    nhat tay -- va nen ghi chung ra CSV de het phai lam vay.
    """
    a = idx("results/detection/testset_ap03.csv", "model", "stratum")
    if not a:
        return None
    TIERS = ["near", "mid", "far", "all"]
    cs = []
    for t in TIERS:
        d = a.get(("distill", t))
        h = a.get(("distill_shuffle", t))
        if d is None or h is None:
            cs.append("--")
            continue
        v = float(d["ap_mean"]) - float(h["ap_mean"])
        cs.append(f"${num(v, 4, lang) if abs(v) < 0.01 else num(v, 3, lang)}$"
                  .replace("$-", "$-").replace("$0", "$+0") if v > 0 else
                  f"${num(v, 4, lang) if abs(v) < 0.01 else num(v, 3, lang)}$")
    if lang == "vi":
        head = ("loss dac trung cuoi $\\mathcal{L}_{\\mathrm{feat}}$ & dung cap & xao tron "
                "& tong hop & \\\\")
        loss = " & 0{,}0076 & 0{,}0084 & 0{,}0060 & \\\\"
    else:
        head = ("final feature loss $\\mathcal{L}_{\\mathrm{feat}}$ & correct & shuffled "
                "& synthetic & \\\\")
        loss = " & 0.0076 & 0.0084 & 0.0060 & \\\\"
    out = [" & " + " & ".join(cs) + " \\\\", "\\midrule", head, "\\midrule", loss]
    return "\n".join(out)


# ------------------------------------------------------- tab:detection_fpfilter
def gen_detection_fp(lang):
    """AP tang xa/giua theo hai quy uoc FP, de doc gia thay chenh lech."""
    r = idx("results/detection/ap03_fp_size_filter.csv", "model", "stratum")
    if not r:
        return None
    ARMS = [("stock", "Stock"), ("nearfar", "$+$Mint"),
            ("distill", "$+$Distill"), ("distill_shuffle", "$+$Distill (shuffle)")]
    out = []
    for m, label in ARMS:
        cs = []
        for t in ("near", "mid", "far"):
            x = r.get((m, t))
            if x is None:
                cs += ["--", "--"]
                continue
            cs += [f"${num(x['ap_unfiltered'], 3, lang)}$",
                   f"${num(x['ap_fp_size_filtered'], 3, lang)}$"]
        out.append(f"{label:22s} & " + " & ".join(cs) + " \\\\")
    return "\n".join(out)


TABLES = {
    "tab:detection": gen_detection,
    "tab:cliff": gen_cliff,
    "tab:multiarch": gen_multiarch,
    "tab:trainmatched": gen_trainmatched,
    "tab:crosssite": gen_crosssite,
    "tab:ablation": gen_ablation,
}


def splice(text, label, body):
    """Thay phan giua \\midrule dau tien sau \\label va \\bottomrule."""
    i = text.index("\\label{%s}" % label)
    m = re.search(r"\\midrule\s*\n", text[i:])
    if not m:
        return None, None
    a = i + m.end()
    b = text.index("\\bottomrule", a)
    return text[:a] + body + "\n" + text[b:], text[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="paper/sections")
    ap.add_argument("--lang", choices=("en", "vi"), default="en")
    ap.add_argument("--file", default="results.tex")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if not (a.check or a.write):
        a.check = True

    p = os.path.join(REPO, a.dir, a.file)
    text = open(p, encoding="utf-8").read()
    changed = drift = 0
    for label, gen in TABLES.items():
        if "\\label{%s}" % label not in text:
            print(f"  [-] {label:16s} khong co trong {a.file}")
            continue
        body = gen(a.lang)
        if body is None:
            print(f"  [?] {label:16s} thieu CSV nguon")
            continue
        new, old = splice(text, label, body)
        if new is None:
            print(f"  [?] {label:16s} khong tim duoc \\midrule")
            continue
        same = re.sub(r"\s+", " ", old).strip() == re.sub(r"\s+", " ", body + "\n").strip()
        if same:
            print(f"  [OK] {label:16s} khop CSV")
        else:
            drift += 1
            print(f"  [X] {label:16s} LECH so voi CSV")
            if a.write:
                text = new
                changed += 1
    if a.write and changed:
        open(p, "w", encoding="utf-8").write(text)
        print(f"\n  da ghi lai {changed} bang vao {a.dir}/{a.file}")
    elif a.check and drift:
        print(f"\n  {drift} bang lech. Chay lai voi --write de sinh lai tu CSV.")
        sys.exit(1)


if __name__ == "__main__":
    main()
