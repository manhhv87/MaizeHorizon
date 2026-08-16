#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dung ban .docx cua ban thao tu nguon LaTeX, cho cac tap chi doi Word.

Biosystems Engineering desk-reject ban nop dau tien mot phan vi dieu nay: he thong
kiem tai lieu tham khao cua ho khong chay duoc tren PDF dung tu LaTeX, nen 56 ref
da xac minh ky thi ho khong xac minh lai duoc.

Lam tren BAN SAO trong thu muc build; nguon LaTeX khong bi dung toi.

Bon phep bien doi, moi cai giai mot van de da do duoc khi thu pandoc trang tron:

  \\fitw{}      Macro bao ca 7 bang la \\resizebox{...}{!}{...}. Pandoc khong hieu
                \\resizebox nen VUT NOI DUNG -- mat sach ca 7 bang. Go lop bao.
  hinh ve inline fig:cdf va fig:pipeline ve bang tikz/pgfplots, pandoc khong dung
                duoc. Thay bang \\includegraphics tro toi PNG da render san.
  duoi anh     \\includegraphics{fig_cliff} khong co duoi -> pandoc bao khong tim
                thay. Chi ro .png (da co san trong figures/).
  \\textsc      Trong pseudocode cua thuat toan, lam vo bo phan tich math cua pandoc.

Danh muc tai lieu tham khao khong di qua pandoc: biblatex+biber da dung san ban APA
dung chuan trong main.pdf, nen boc thang tu do ra con dung hon la de pandoc dung lai
bang citeproc voi mot CSL khac.

    python scripts/build_docx.py
    python scripts/build_docx.py --out /duong/dan/manuscript.docx
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper")


def strip_braced(s, cmd):
    """Bo \\cmd{ } giu lai RUOT, co khop ngoac long nhau."""
    out, i = [], 0
    while True:
        j = s.find("\\" + cmd + "{", i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k = j + len(cmd) + 2
        d = 1
        start = k
        while k < len(s) and d:
            if s[k] == "{":
                d += 1
            elif s[k] == "}":
                d -= 1
            k += 1
        out.append(s[start:k - 1])
        i = k


def replace_float(t, label, png):
    """Thay ca khoi figure ve bang tikz bang mot \\includegraphics."""
    i = t.find("\\label{" + label + "}")
    if i < 0:
        return t, False
    s = t.rfind("\\begin{figure", 0, i)
    e = t.find("\\end{figure", i)
    if s < 0 or e < 0:
        return t, False
    blk = t[s:e]
    m = re.search(r"\\caption\{", blk)
    cap = ""
    if m:
        k, d = m.end(), 1
        while k < len(blk) and d:
            if blk[k] == "{":
                d += 1
            elif blk[k] == "}":
                d -= 1
            k += 1
        cap = blk[m.end():k - 1]
    new = ("\\begin{figure}[!ht]\n\\centering\n"
           f"\\includegraphics[width=\\linewidth]{{{png}}}\n"
           f"\\caption{{{cap}}}\n\\label{{{label}}}\n")
    return t[:s] + new + t[e:], True


# Danh muc tai lieu tham khao: boc tu main.pdf (biblatex+biber da dung san ban APA)
# thay vi de pandoc dung lai bang citeproc voi mot CSL khac.
_PART = r"(?:de|la|le|van|von|der|del|di|da|dos|el|y|Van|De|La)"
_START = re.compile(rf"^(?:{_PART}\s+)*[A-ZÀ-ÝĐ][\w'’\-]*"
                    rf"(?:\s+(?:{_PART}|[A-ZÀ-ÝĐ][\w'’\-]*))*,\s+[A-Z]\.")


def extract_references(pdf):
    """Ghep lai danh muc tu PDF hai cot dong doi + co so dong.

    Ba cho da vap khi lam:
      - \linenumbers chen so dong vao giua -> bo dong chi co chu so
      - ho co tieu tu viet thuong o dau HOAC o giua (de Silva, Sanchez de la
        Fuente) -> mau bat dau phai cho phep ca hai
      - dong ket thuc bang gach noi: co the la ngat tu (bo gach) hoac la gach
        cua chinh ten ghep (giu gach). Phan biet bang chu cai dau dong sau:
        hoa -> ten ghep, thuong -> ngat tu.
    """
    txt = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
    i = txt.rfind("\nReferences")
    if i < 0:
        return []
    lines = [s.strip() for s in txt[i + 11:].split("\n")
             if s.strip() and not re.fullmatch(r"\d{1,4}", s.strip())]
    ents = []
    for s in lines:
        if ents and ents[-1].endswith("-"):
            p = ents[-1]
            ents[-1] = (p + s) if s[:1].isupper() else (p[:-1] + s)
            continue
        if _START.match(s) and (not ents or len(ents[-1]) > 60):
            ents.append(s)
        elif ents:
            ents[-1] += " " + s
        else:
            ents.append(s)
    return [re.sub(r"\s+", " ", e).strip() for e in ents if len(e) > 50]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", default=os.path.join(REPO, "_docx_build"))
    ap.add_argument("--out", default=os.path.join(PAPER, "manuscript.docx"))
    ap.add_argument("--figs", default=os.path.join(REPO, "_docx_build", "figs"),
                    help="thu muc chua fig_cdf.png va fig_pipeline.png da render")
    a = ap.parse_args()

    if os.path.isdir(a.build):
        shutil.rmtree(a.build)
    shutil.copytree(PAPER, a.build, ignore=shutil.ignore_patterns(
        "*.pdf", "*.aux", "*.log", "*.bbl", "*.bcf", "*.out", "*.fls",
        "*.fdb_latexmk", "*.run.xml", "*.synctex.gz", "_docx_build"))
    os.makedirs(a.figs, exist_ok=True)

    changed = {"fitw": 0, "ext": 0, "textsc": 0, "float": 0}
    for f in glob.glob(os.path.join(a.build, "sections", "*.tex")):
        t = open(f, encoding="utf-8").read()
        o = t
        n = t.count("\\fitw{")
        t = strip_braced(t, "fitw")
        changed["fitw"] += n
        for stem in ("fig_cliff", "fig_scaling"):
            k = t.count("{" + stem + "}")
            t = t.replace("{" + stem + "}", "{figures/" + stem + ".png}")
            changed["ext"] += k
        t = t.replace("{maizehorizon_example.jpg}", "{figures/maizehorizon_example.jpg}")
        k = t.count("\\textsc{")
        t = strip_braced(t, "textsc")
        changed["textsc"] += k
        for lab, png in (("fig:cdf", "figs/fig_cdf.png"),
                         ("fig:pipeline", "figs/fig_pipeline.png")):
            t, ok = replace_float(t, lab, png)
            changed["float"] += int(ok)
        if t != o:
            open(f, "w", encoding="utf-8").write(t)

    for p in glob.glob(os.path.join(a.figs, "*.png")):
        shutil.copy(p, os.path.join(a.build, "figs", os.path.basename(p))
                    if os.path.isdir(os.path.join(a.build, "figs"))
                    else a.build)
    os.makedirs(os.path.join(a.build, "figs"), exist_ok=True)
    for p in glob.glob(os.path.join(a.figs, "*.png")):
        shutil.copy(p, os.path.join(a.build, "figs"))

    print("  bien doi:", ", ".join(f"{k}={v}" for k, v in changed.items()))

    # Tiem danh muc vao ban dung TRUOC pandoc: ra mot file Word lien mach thay vi
    # de nguoi dung dan tay. \printbibliography khong chay duoc ngoai LaTeX nen
    # bi thay bang cac doan van thuong; Word se dinh dang lai duoc.
    refs = extract_references(os.path.join(PAPER, "main.pdf"))
    if refs:
        blk = ["\\section*{References}", ""]
        for e in refs:
            blk.append(e.replace("\\", "\\textbackslash{}").replace("&", "\\&")
                        .replace("%", "\\%").replace("_", "\\_").replace("#", "\\#"))
            blk.append("")
        mt = os.path.join(a.build, "main.tex")
        s = open(mt, encoding="utf-8").read()
        s = re.sub(r"\\printbibliography(\[[^\]]*\])?", lambda _m: "\n".join(blk), s)
        open(mt, "w", encoding="utf-8").write(s)
        print(f"  danh muc: tiem {len(refs)} muc vao ban dung")
    else:
        print("  CANH BAO: khong boc duoc danh muc tu main.pdf")

    cmd = ["pandoc", "main.tex", "-o", a.out,
           "--resource-path=.:figures:figs", "--from", "latex", "--to", "docx"]
    r = subprocess.run(cmd, cwd=a.build, capture_output=True, text=True)
    warn = [l for l in r.stderr.split("\n") if l.startswith("[WARNING]")]
    print(f"  pandoc exit={r.returncode}, {len(warn)} canh bao")
    for w in warn[:6]:
        print("   ", w[:110])


    print(f"\n-> {a.out}")
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
