#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Moi CSV trong results/ phai truy duoc ve mot lenh cu the. Script nay kiem dieu do.

  python check_traceability.py            # bao cao
  python check_traceability.py --strict   # exit 1 neu con CSV mo coi

Quy tac cua repo: cai gi chay ra so thi phai co script sinh ra no, va phai co cho ghi
lai lenh da chay. Neu mot CSV khong truy duoc ve lenh nao thi khong ai lap lai duoc no,
va con so trong bai dua tren no khong kiem chung duoc.

Nguon "lenh da chay" duoc coi la hop le:
  - rerun_after_relabel.py   (nguon su that cho duong tai lap chinh)
  - docstring / gia tri --out mac dinh trong cac script *.py o thu muc goc
  - cac khoi ```bash trong results/**/README.md

Bon kieu khop, vi script khong phai luc nao cung nhan nguyen ten file:
  exact    ten file xuat hien nguyen ven
  prefix   sinh boi --out-prefix P; CSV la P + hau to (_pot/_phys/_range/_prec/...)
  derived  script tu noi hau to vao --out (vd --out X.csv -> X_curve.csv, X_eval.csv)
  snapshot ban sao trong results/*_snapshot/, truy theo ban goc
  stem     moi manh cua ten file deu xuat hien o dau do trong mot file

Canh bao ve `stem`: no la khop YEU va de bao nham. `count_gates.csv` tung duoc no cho
qua chi vi ca "count" lan "gates" deu xuat hien trong furrowmap_count.py, trong khi
script do khong he ghi CSV (ket qua chi ra stdout, bang phai ghep tay). Vi vay `--strict`
coi `stem` la CHUA truy duoc; dung `--allow-stem` neu muon giu hanh vi cu.
"""
import argparse
import glob
import os
import re

# Hau to ma cac script tu noi them, khong bao gio xuat hien trong lenh.
PREFIX_SUFFIXES = ("_pot", "_phys", "_physprec", "_prec", "_range")
DERIVED_SUFFIXES = ("_curve", "_eval", "_eval_curve")


def load_sources():
    """Moi cho co the ghi lai mot lenh: script goc + README trong results/."""
    src = {}
    for f in sorted(glob.glob("*.py")):
        src[f] = open(f, encoding="utf-8", errors="replace").read()
    for f in sorted(glob.glob("results/**/*.md", recursive=True)):
        src[f] = open(f, encoding="utf-8", errors="replace").read()
    return src


def out_prefixes(src):
    """Moi gia tri --out-prefix xuat hien o bat ky dau."""
    pref = set()
    for t in src.values():
        for m in re.finditer(r"--out-prefix[=\s\"']+([\w./\\-]+)", t):
            pref.add(m.group(1).replace("\\", "/"))
        # dang argv trong rerun_after_relabel.py: "--out-prefix", f"results/..."
        for m in re.finditer(r'"--out-prefix",\s*f?"([^"]+)"', t):
            pref.add(re.sub(r"\{[^}]*\}", "", m.group(1)))
    return pref


def classify(csv, src, pref):
    """(kieu_khop, cho_tim_thay) hoac (None, None) neu mo coi."""
    base = os.path.basename(csv)
    stem = base[:-4]
    norm = csv.replace("\\", "/")

    for f, t in src.items():
        if norm in t or base in t:
            return "exact", f

    for p in pref:
        if not p:
            continue
        if norm.startswith(p) and any(stem.endswith(s) for s in PREFIX_SUFFIXES):
            for f, t in src.items():
                if p in t:
                    return "prefix", f

    for suf in DERIVED_SUFFIXES:
        if stem.endswith(suf):
            root = stem[: -len(suf)] + ".csv"
            for f, t in src.items():
                if root in t or os.path.basename(root) in t:
                    return "derived", f

    # Ban chup: results/<snap>/X la ban sao cua results/X, nen no truy duoc khi ban goc
    # truy duoc. README cua thu muc chup giai thich quan he do.
    parts_path = norm.split("/")
    if len(parts_path) > 2 and parts_path[1].endswith("_snapshot"):
        orig = "/".join([parts_path[0]] + parts_path[2:])
        k, w = classify(orig, src, pref)
        if k:
            return "snapshot", f"ban goc {orig} <- {w}"

    # Truong hop cuoi: ten bi cat thanh nhieu manh trong script (f-string, noi chuoi).
    parts = [x for x in re.split(r"[_.]", stem) if len(x) > 3]
    if parts:
        for f, t in src.items():
            if all(p in t for p in parts):
                return "stem", f
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 neu con CSV mo coi; khop 'stem' cung bi coi la mo coi")
    ap.add_argument("--allow-stem", action="store_true",
                    help="chap nhan khop 'stem' (yeu) la da truy duoc")
    ap.add_argument("--verbose", action="store_true", help="in ca cac CSV da truy duoc")
    a = ap.parse_args()

    src = load_sources()
    pref = out_prefixes(src)
    csvs = sorted(glob.glob(os.path.join(a.dir, "**", "*.csv"), recursive=True))
    if not csvs:
        raise SystemExit(f"khong thay CSV nao trong {a.dir}/")

    kinds, orphan = {}, []
    for c in csvs:
        k, where = classify(c.replace("\\", "/"), src, pref)
        if k is None:
            orphan.append(c)
        else:
            kinds.setdefault(k, []).append((c, where))

    print(f"{len(csvs)} CSV trong {a.dir}/")
    for k in ("exact", "prefix", "derived", "snapshot", "stem"):
        if k in kinds:
            print(f"  {k:8s} {len(kinds[k]):3d}")
            if a.verbose:
                for c, w in kinds[k]:
                    print(f"           {c}  <- {w}")
    weak = kinds.get("stem", []) if (a.strict and not a.allow_stem) else []
    if weak:
        print(f"\n  {len(weak)} khop 'stem' -- YEU, --strict coi la chua truy duoc:")
        for c, w in weak:
            print(f"           {c}  <- doan la {w}")
    print(f"  {'MO COI':8s} {len(orphan):3d}")
    for c in orphan:
        print(f"           {c}")

    if orphan:
        print("\nMoi CSV mo coi la mot cho trong bai khong kiem chung duoc.")
        print("Sua bang cach ghi lenh da chay vao results/<nhom>/README.md,")
        print("hoac vao docstring cua script sinh ra no.")
    if a.strict and (orphan or weak):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
