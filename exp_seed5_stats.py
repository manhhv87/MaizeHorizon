#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phep kiem cho phep dao dau tiling, tinh lai tu gia tri tung seed.

Bai bao cao ba con so o Muc res-tiling: muc giam tren bo 1080p, muc tang tren bo 8K,
va hieu giua hai muc do (phep kiem tuong tac). Ca ba deu duoc tinh o n=3 va deu duoc
lay tu bang tong hop chi co mean/std. Script nay tinh lai tu AP tho tung seed do
exp_sahi_baseline.py --per-seed-out ghi ra.

  python exp_seed5_stats.py \
      --p1080 results/baselines/sahi_perseed_1080p_n5.csv \
      --p8k   results/crosssite/hd_sahi_perseed_n5.csv \
      --out   results/rebuttal/tiling_reversal_stats.csv

Mot diem ve phuong phap. Ban dang co trong bai dung Welch KHONG ghep cap giua hai
nhanh full va sahi. Nhung hai con so ay den tu CUNG mot checkpoint, chi khac cach
suy luan, nen chung ghep cap tu nhien: seed nao manh o full thi cung manh o sahi.
Bo qua su ghep cap ay lam mat luc kiem dinh ma khong duoc gi. Vi vay script bao cao
ca hai:

  paired  -- kiem t mot mau tren hieu theo tung seed  (nen dung, manh hon)
  welch   -- kiem t Welch hai mau khong ghep cap      (giu lai de doi chieu voi ban cu)

Phep kiem tuong tac la Welch hai mau tren HIEU cua hai bo du lieu, vi hai bo khong
chia se seed theo bat ky nghia nao co the ghep cap duoc.
"""
import argparse
import csv
import math
import os


def read_deltas(path, tier="far"):
    """[(seed, full, sahi, delta)] cho mot tang."""
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["tier"] == tier:
            out.append((r["seed"], float(r["full_ap"]), float(r["sahi_ap"]), float(r["delta"])))
    if not out:
        raise SystemExit(f"khong thay tang {tier} trong {path}")
    return out


def mean_sd(v):
    n = len(v)
    m = sum(v) / n
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, s, n


def paired_t(deltas):
    """Kiem t mot mau tren hieu theo tung seed."""
    from scipy import stats
    m, s, n = mean_sd(deltas)
    se = s / math.sqrt(n)
    t, p = stats.ttest_1samp(deltas, 0.0)
    return dict(diff=m, se=se, t=float(t), df=n - 1, p=float(p), n=n)


def welch_t(a, b):
    """Welch hai mau khong ghep cap: b - a."""
    from scipy import stats
    ma, sa, na = mean_sd(a)
    mb, sb, nb = mean_sd(b)
    se = math.sqrt(sa * sa / na + sb * sb / nb)
    r = stats.ttest_ind(b, a, equal_var=False)
    return dict(diff=mb - ma, se=se, t=float(r.statistic), df=float(r.df),
                p=float(r.pvalue), n=min(na, nb))


def paired_interaction(a, b):
    """Hieu-cua-hieu, ghep cap theo seed: a - b tren tung seed."""
    from scipy import stats
    d = [x - y for x, y in zip(a, b)]
    m, sd, n = mean_sd(d)
    r = stats.ttest_rel(a, b)
    return dict(diff=m, se=sd / math.sqrt(n), t=float(r.statistic),
                df=float(n - 1), p=float(r.pvalue), n=n)


def fmt(d):
    return (f"diff={d['diff']:+.4f}  SE={d['se']:.4f}  t={d['t']:+.2f}  "
            f"df={d['df']:.1f}  p={d['p']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1080", required=True, help="per-seed CSV cua bo 1080p")
    ap.add_argument("--p8k", required=True, help="per-seed CSV cua bo 8K")
    ap.add_argument("--tier", default="far")
    ap.add_argument("--label-1080", default="1080p (cam bien rang buoc)")
    ap.add_argument("--label-8k", default="8K (dau vao rang buoc)")
    ap.add_argument("--out", default="results/rebuttal/tiling_reversal_stats.csv")
    a = ap.parse_args()

    rows = [["corpus", "test", "n_seeds", "full_ap_mean", "sahi_ap_mean",
             "diff", "se", "t", "df", "p"]]
    keep, keep_seeds = {}, {}
    for path, lab in ((a.p1080, a.label_1080), (a.p8k, a.label_8k)):
        d = read_deltas(path, a.tier)
        full = [x[1] for x in d]
        sahi = [x[2] for x in d]
        dl = [x[3] for x in d]
        keep[lab] = dl
        keep_seeds[lab] = [x[0] for x in d]
        fm = sum(full) / len(full)
        sm = sum(sahi) / len(sahi)
        print(f"\n=== {lab} | tang {a.tier} | {len(d)} seed ===")
        print(f"  full AP  {fm:.4f}   sahi AP  {sm:.4f}   thay doi tuong doi "
              f"{100*(sm-fm)/fm:+.0f}%")
        for name, res in (("paired", paired_t(dl)), ("welch", welch_t(full, sahi))):
            print(f"  {name:7s} {fmt(res)}")
            rows.append([lab, name, len(d), round(fm, 6), round(sm, 6),
                         round(res["diff"], 6), round(res["se"], 6),
                         round(res["t"], 3), round(res["df"], 2), round(res["p"], 4)])

    labs = list(keep)
    # Ghep cap khi va chi khi hai ve chay tren CUNG tap seed, tuc cung checkpoint.
    # Doi chung E5 thoa dieu do (ca hai nhanh la stock_s0..s4 tren cung canh); phep
    # so 1080p<->8K thi khong (nearfar so voi stock, hai model khac nhau), nen o do
    # Welch moi dung. Bo phan ghep cap khi co no la vut bo luc thong ke: tren E5,
    # ghep cap cho t(4)=4,80 p=0,0086 con Welch cho t=3,85 p=0,0122.
    sa, sb = keep_seeds[labs[0]], keep_seeds[labs[1]]
    if sa and sa == sb:
        inter = paired_interaction(keep[labs[1]], keep[labs[0]])
        how = "paired_on_deltas"
    else:
        inter = welch_t(keep[labs[0]], keep[labs[1]])
        how = "welch_on_deltas"
    print(f"\n=== phep kiem tuong tac: hieu giua hai muc thay doi ({how}) ===")
    print(f"  {fmt(inter)}")
    if how == "paired_on_deltas":
        print("  (ghep cap: hai ve dung chung checkpoint va chung canh)")
    else:
        print("  (Welch: hai ve dung nhanh/checkpoint khac nhau, khong ghep cap duoc)")
    print("  (day moi la khang dinh cua bai: dau doi chieu, khong phai do lon tung nua)")
    rows.append(["interaction", how, inter["n"], "", "",
                 round(inter["diff"], 6), round(inter["se"], 6),
                 round(inter["t"], 3), round(inter["df"], 2), round(inter["p"], 4)])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
