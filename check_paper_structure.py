#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soat ky luat IMRaD cua ban thao: noi dung co nam dung phan cua no khong.

Khong cham khoa hoc. Chi tra loi mot cau hoi lap di lap lai trong review:
*cai nay co thuoc ve day khong*, va bat cac loi hinh thuc di kem.

Chay:
    python check_paper_structure.py                 # ban EN
    python check_paper_structure.py --dir paper/vi/sections --lang vi

Ma thoat 1 neu co phat hien o muc HIGH, de cam duoc vao pre-commit hoac CI.

Bay phep kiem, moi phep ung voi mot loi da thuc su xay ra trong ban thao nay:

  1. intro-numbers    So lieu KET QUA trong Introduction. Khong tinh so cua cong
                      trinh duoc trich (cau co \parencite/\cite) va khong tinh
                      nam. Introduction bi bat 22% noi dung la results/method.
  2. intro-forward    Introduction tro toi Results. Doc gia gap ket luan truoc
                      khi gap bang chung.
  3. methods-forward  Methods bien minh thiet ke bang KET QUA chua trinh bay.
                      Chi bat cac dong tu khang dinh (shows/confirms/demonstrates),
                      khong bat chi duong ("results are in ...").
  4. methods-limits   Ngon ngu han che/nhan dinh nam trong Methods.
  5. results-verdict  Cau quy ket / khuyen nghi trien khai nam trong Results.
  6. stub-para        Doan 1-2 cau lac long (thuong la tan du cua mot lan cat).
  7. dup-word         Lap tu lien tiep ("Second, Second,").
"""
import argparse
import os
import re
import sys

HIGH, MED, LOW = "HIGH", "MED", "LOW"

# Cau khang dinh mot ket qua chua trinh bay, khac han voi chi duong sang no.
ASSERTIVE = r"(?:shows?|confirms?|demonstrates?|establishes?|proves?|reveals?)"
# Ngon ngu han che: thuoc Discussion, khong thuoc Methods.
LIMIT_CUES = [
    r"we (?:did not|do not|have not) (?:run|do|test|measure|attempt)",
    r"no formal .{0,30}study",
    r"should be read as",
    r"is thus a limitation",
    r"rather than a (?:general|measured)",
]
# Quy ket / khuyen nghi: thuoc Discussion, khong thuoc Results.
VERDICT_CUES = [
    r"the honest measure",
    r"makes it the better way",
    r"we (?:would|should) deploy",
    r"is a design constraint",
    r"argues for reporting",
    r"the useful statement is",
]


def paragraphs(text):
    """Tach doan, bo comment va moi truong float (bang/hinh/thuat toan)."""
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    for env in ("table", "figure", "algorithm", "algorithmic", "itemize", "equation"):
        text = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env), " ", text, flags=re.S)
    out = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if p and not p.startswith(("\\section", "\\subsection", "\\subsubsection", "\\label", "\\input")):
            out.append(p)
    return out


def paragraphs_flagged(text):
    """Tach doan, kem hai co: doan mo mot muc con, va doan ke mot float.

    Ca hai deu la ly do chinh dang khien mot doan ngan: cau mo muc, va cau dan
    vao / chot lai mot bang hay hinh. Float duoc thay bang mot moc thay vi bi
    xoa, de con biet doan nao ke no.
    """
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    for env in ("table", "figure", "algorithm", "algorithmic", "itemize", "equation"):
        text = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env),
                      "\n\n<<FLOAT>>\n\n", text, flags=re.S)
    raw = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if not p:
            continue
        if p.startswith(("\\section", "\\subsection", "\\subsubsection")):
            raw.append(("HEAD", p)); continue
        if p.startswith(("\\label", "\\input")):
            continue
        raw.append(("FLOAT" if p == "<<FLOAT>>" else "PARA", p))

    out, lead = [], True
    for k, (kind, p) in enumerate(raw):
        if kind == "HEAD":
            lead = True; continue
        if kind == "FLOAT":
            continue
        prev = raw[k - 1][0] if k else None
        nxt = raw[k + 1][0] if k + 1 < len(raw) else None
        near_float = prev == "FLOAT" or nxt == "FLOAT"
        out.append((lead or near_float, p))
        lead = False
    return out


def strip_cited(p):
    """Bo cac cau co trich dan: so trong do la cua nguoi khac, khong phai cua bai."""
    return " ".join(s for s in re.split(r"(?<=\.)\s+", p)
                    if not re.search(r"\\(?:paren|foot|text)?cite", s))


def check(d, lang):
    def read(name):
        fp = os.path.join(d, name + ".tex")
        return open(fp, encoding="utf-8").read() if os.path.exists(fp) else ""

    intro, meth = read("introduction"), read("materials_methods")
    res, disc = read("results"), read("discussion")
    found = []

    def add(sev, rule, where, msg):
        found.append((sev, rule, where, msg))

    # 1 -- so lieu ket qua trong Introduction
    for p in paragraphs(intro):
        for m in re.finditer(r"\$[^$]*\d[^$]*\$", strip_cited(p)):
            tok = m.group(0)
            if re.fullmatch(r"\$\d{4}\$", tok):       # nam
                continue
            add(HIGH, "intro-numbers", "introduction", f"so lieu {tok} — Introduction khong nen mang so ket qua")

    # 2 -- Introduction tro toi Results
    for m in re.finditer(r"\\ref\{(sec:res-[^}]+)\}", intro):
        add(HIGH, "intro-forward", "introduction", f"tro toi {m.group(1)}")

    # 3 -- Methods bien minh bang ket qua
    for m in re.finditer(r"Section~\\ref\{sec:res-[^}]+\}\s+" + ASSERTIVE, meth):
        add(HIGH, "methods-forward", "materials_methods", f"'...{m.group(0)[-52:]}' — thiet ke vien ket qua chua co")

    # 4 -- ngon ngu han che trong Methods
    for cue in LIMIT_CUES:
        for m in re.finditer(cue, meth, re.I):
            a = max(0, m.start() - 60)
            add(MED, "methods-limits", "materials_methods",
                "..." + re.sub(r"\s+", " ", meth[a:m.end() + 30]).strip()[-92:])

    # 5 -- quy ket trong Results
    for cue in VERDICT_CUES:
        for m in re.finditer(cue, res, re.I):
            a = max(0, m.start() - 60)
            add(MED, "results-verdict", "results",
                "..." + re.sub(r"\s+", " ", res[a:m.end() + 30]).strip()[-92:])

    # 6 -- doan cut.
    # Ngan KHONG phai la loi: cau mo mot muc con va ghi chu ha tang ngan la co ly
    # do. Loi la mot MENH DE LE bi bo lai sau mot lan cat -- mot cau, duoi 25 tu,
    # va khong ke mot bang/hinh (cau dan vao hoac chot lai float thi ngan la dung).
    for name, txt in (("introduction", intro), ("materials_methods", meth),
                      ("results", res), ("discussion", disc)):
        for is_lead, p in paragraphs_flagged(txt):
            n = len(p.split())
            if is_lead or not n:
                continue
            if n < 25 and len(re.findall(r"\.\s", p)) + 1 == 1:
                add(LOW, "stub-para", name, f"{n} tu: {re.sub(chr(92)+'s+',' ',p)[:66]}...")

    # 7 -- lap tu
    # Chi mien cho nhung tu ma lap lai LA hop le trong tieng Anh: "had had" (qua khu
    # hoan thanh) va "that that" ("the fact that that happened"). Moi tu khac lap lai
    # deu la loi go -- ke ca "the the", von bi danh sach cu nuot mat.
    stop = {"that", "had"}
    for name, txt in (("introduction", intro), ("materials_methods", meth),
                      ("results", res), ("discussion", disc)):
        # Dau phay co the xen vao giua: loi that trong ban thao la "Second,  Second,".
        # KHONG nhan dau cham phay hay hai cham: chung danh dau ranh gioi menh de,
        # noi lap lai mot tu la thu phap ("...for 50% recall; recall declines...").
        for m in re.finditer(r"\b([A-Za-z]{3,}),?\s+\1\b", re.sub(r"(?m)^\s*%.*$", "", txt)):
            if m.group(1).lower() not in stop:
                add(HIGH, "dup-word", name, f"lap tu: '{m.group(0)}'")

    # 8 -- dau hieu van phong may
    LEX = ("delve", "showcase", "underscore", "pivotal", "realm", "tapestry",
           "holistic", "multifaceted", "seamless", "intricate", "paradigm",
           "testament", "myriad", "plethora", "harness", "embark", "meticulous")
    PHR = (r"[Ii]t is (?:worth noting|important to note|crucial to note)",
           r"[Ii]n the realm of", r"[Ss]hedding light", r"[Aa] deep dive",
           r"plays? an? (?:crucial|vital|key) role", r"[Bb]y leveraging")
    for name, txt in (("introduction", intro), ("materials_methods", meth),
                      ("results", res), ("discussion", disc)):
        clean = re.sub(r"(?m)^\s*%.*$", "", txt)
        for w in LEX:
            for m in re.finditer(r"\b" + w + r"\b", clean, re.I):
                add(MED, "ai-tells", name, f"tu rong: '{m.group(0)}'")
        for pat in PHR:
            for m in re.finditer(pat, clean):
                add(MED, "ai-tells", name, f"cum mau: '{m.group(0)}'")
        # Ky hieu: em-dash "---" trong than bai. Dung dung thi vo hai, nhung no la
        # dau hieu bi soi nhat, va o day moi cho dung deu thay bang dau phay, hai
        # cham, ngoac don hoac tach cau ma khong mat nghia. En-dash "--" trong dai
        # so ($0.28$--$0.34$) la kieu chu dung, khong bat.
        for m in re.finditer(r"(?<!-)---(?!-)", clean):
            a = max(0, m.start() - 40)
            add(MED, "ai-tells", name,
                "em-dash: ..." + re.sub(r"\s+", " ", clean[a:m.end() + 34]).strip())

        # Tat nhip: mot cau dung hai lan cung mot cau truc. Bo nhan in dam truoc
        # khi dem: dau hai cham trong "\textbf{C3: ...}" la nhan de muc, khong
        # phai nhip van, va dem ca no thi moi muc dong gop deu bi bao nham.
        prose = re.sub(r"\\textbf\{[^}]*\}", " ", re.sub(r"\s+", " ", clean))
        for sent in re.split(r"(?<=[.]) ", prose):
            # Dem CA HO cau truc doi lap, khong rieng "rather than": neu chi
            # bat mot bien the thi viec sua chi day tat sang bien the khac
            # (dung nhu da xay ra: 49 -> 20 nhung ", not X" tang tu 30 len 50).
            for tic in (r"\brather than\b", r",\s+not\s+\w", r"\binstead of\b", r"\w: [a-z]"):
                if len(re.findall(tic, sent)) >= 2:
                    add(LOW, "ai-tells", name, f"lap cau truc x2 trong 1 cau: ...{sent.strip()[:66]}")

    # 9 -- mat do dau cau. Sua mot tat ma khong do ca ho thi chi day tat sang cho
    # khac: da xay ra hai lan, "rather than" -> ", not X", roi em-dash -> hai cham.
    # Nguong dat theo muc da dat duoc sau khi don, cong bien do.
    body = " ".join(re.sub(r"(?m)^\s*%.*$", "", x) for x in (intro, meth, res, disc))
    for e in ("table", "figure", "algorithm", "algorithmic", "equation"):
        body = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (e, e), " ", body, flags=re.S)
    body = re.sub(r"\\textbf\{[^}]*\}", " ", body)
    W = max(len(body.split()), 1)
    for label, pat, cap in (("hai cham giua cau", r"[a-z0-9\}\$]: [a-z]", 4.5),
                            ("dau cham phay",     r";",                      7.0),
                            ("doi lap 'rather than'", r"\brather than\b",   2.5),
                            ("doi lap ', not X'",  r",\s+not\s+\w",         5.0),
                            ("em-dash",            r"(?<!-)---(?!-)",        0.5)):
        dens = 1000.0 * len(re.findall(pat, body)) / W
        if dens > cap:
            add(MED, "ai-tells", "than bai",
                f"mat do {label}: {dens:.1f}/1000 tu (nguong {cap})")

    # 10 -- nhip cua Abstract. Day la phan bi soi nhat, va dau hieu manh nhat khong
    # phai tu ngu ma la do DEU cua chieu dai cau: van may co he so bien thien thap
    # (do ~0.34 truoc khi sua; van nguoi thuong 0.5-0.7). Cung bat cac the doi xung
    # hay gap: cap "On a ... On a ...", bo ba "two X, two Y and two Z".
    abst = read("abstract")
    if abst:
        a = re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+\{?|[{}$~\\]", " ", abst))
        a = a[:a.index("Keywords")] if "Keywords" in a else a
        sents = [x for x in re.split(r"(?<=[.]) ", a.strip()) if len(x.split()) > 2]
        L = [len(x.split()) for x in sents]
        if len(L) >= 4:
            mean = sum(L) / len(L)
            sd = (sum((x - mean) ** 2 for x in L) / len(L)) ** 0.5
            if mean and sd / mean < 0.45:
                add(MED, "ai-tells", "abstract",
                    f"chieu cau qua deu: he so bien thien {sd/mean:.2f} (nen >= 0.45)")
        for label, pat in (("cap doi xung 'On a ... On a ...'", r"On a [^.]+\. On a "),
                           ("bo ba 'two X, two Y and two Z'", r"two \w+, two \w+ \w+ and two"),
                           ("cau ket dang cach ngon", r"is therefore [^.]+, and is best")):
            if re.search(pat, a):
                add(MED, "ai-tells", "abstract", label)

    # 11 -- macro \ref bi hong ma KHONG sinh "??".
    # Da xay ra that: mot lenh re.sub co chuoi thay the chua "\\ref" bi Python doc
    # "\\r" thanh ky tu xuong dong, nen "Section~\\ref{x}" thanh "Section~" + newline
    # + "ef{x}". LaTeX in ra "Section ef{x}" nhu van ban thuong: khong "??", khong
    # canh bao, va phep kiem "?? == 0" hoan toan mu truoc no.
    for name, txt in (("introduction", intro), ("materials_methods", meth),
                      ("results", res), ("discussion", disc)):
        for m in re.finditer(r"(?m)^\s*(?:ef|ref)\{", txt):
            add(HIGH, "broken-ref", name, f"macro ref mat dau gach cheo: '{txt[m.start():m.start()+26].strip()}'")
        for m in re.finditer(r"\\\\ref\{", txt):
            add(HIGH, "broken-ref", name, "hai dau gach cheo truoc ref (\\\\ref) -> LaTeX doc thanh ngat dong")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="paper/sections")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args()
    found = check(a.dir, a.lang)
    order = {HIGH: 0, MED: 1, LOW: 2}
    found.sort(key=lambda r: (order[r[0]], r[1]))
    if not found:
        print("[OK] khong phat hien gi.")
        return 0
    cur = None
    for sev, rule, where, msg in found:
        if (sev, rule) != cur:
            cur = (sev, rule)
            print(f"\n[{sev}] {rule}")
        print(f"   {where}: {msg}")
    n_high = sum(1 for f in found if f[0] == HIGH)
    print(f"\n{len(found)} phat hien ({n_high} HIGH)")
    return 1 if n_high else 0


if __name__ == "__main__":
    sys.exit(main())
