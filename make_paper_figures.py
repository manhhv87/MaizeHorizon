#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the .dat files behind the pgfplots figures.

  paper/data/cdf_boxarea.dat   box-area CDF for plant boxes
  paper/data/recall_vs_pot.dat recall vs POT per arm, from an eval *_curve.csv

Also prints the key fractions so the figure caption can be checked.

  python make_paper_figures.py --labels-dir data/test/labels --out paper/data
"""
import argparse, os, glob, csv, sys
import xml.etree.ElementTree as ET


def plant_areas(xml_path):
    """Plant-box areas as a percentage of image area, from a CVAT xml."""
    if not xml_path or not os.path.exists(xml_path):
        return []
    root = ET.parse(xml_path).getroot()
    out = []
    for img in root.iter("image"):
        W = float(img.get("width", 0)) or 1920.0
        H = float(img.get("height", 0)) or 1080.0
        for b in img.iter("box"):
            if (b.get("label", "").lower() != "plant"):
                continue
            x1, y1 = float(b.get("xtl")), float(b.get("ytl"))
            x2, y2 = float(b.get("xbr")), float(b.get("ybr"))
            out.append(100.0 * abs((x2 - x1) * (y2 - y1)) / (W * H))   # percent of image area
    return out


def areas_from_labels(labels_dir, plant_cls=0):
    """Plant-box areas (% of image area) from a YOLO label directory.

    This is the source for the box-area CDF; annotations.xml is a pilot export and gives
    different numbers."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        for line in open(p, encoding="utf-8"):
            s = line.split()
            if len(s) < 5 or int(float(s[0])) != plant_cls:
                continue
            out.append(100.0 * float(s[3]) * float(s[4]))
    return out


def write_cdf(areas, out_dir, npts=28, xmin=0.001, xmax=10.0):
    """Write cdf_boxarea.dat, log-spaced over a fixed [xmin, xmax] range so the x axis is stable."""
    areas = sorted(a for a in areas if a > 0)
    n = len(areas)
    if n == 0:
        return None
    lo, hi = xmin, xmax
    xs = [lo * (hi / lo) ** (i / (npts - 1)) for i in range(npts)]
    rows = []
    j = 0
    for x in xs:
        while j < n and areas[j] <= x:
            j += 1
        rows.append((x, j / n))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cdf_boxarea.dat"), "w", encoding="utf-8") as f:
        f.write("areapct cum\n")
        for x, c in rows:
            f.write(f"{x:.6g} {c:.5f}\n")
    sub01 = 100.0 * sum(1 for a in areas if a < 0.1) / n
    sub05 = 100.0 * sum(1 for a in areas if a < 0.5) / n
    return n, sub01, sub05


def write_recall_curve(curve_csv, out_dir):
    """Pivot an eval *_curve.csv into recall_vs_pot.dat, one recall column per arm."""
    if not curve_csv or not os.path.exists(curve_csv):
        return None
    rows = list(csv.DictReader(open(curve_csv, encoding="utf-8")))
    if not rows:
        return None
    arms, data = [], {}
    for r in rows:
        m = r["model"]
        if m not in arms:
            arms.append(m)
        try:
            hi = float(r["pot_hi_px"]) if r["pot_hi_px"] not in ("inf", "") else 200.0
            rec = float(r["recall_mean"]) if r["recall_mean"] != "" else float("nan")
        except Exception:
            continue
        data.setdefault(hi, {})[m] = rec
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "recall_vs_pot.dat"), "w", encoding="utf-8") as f:
        f.write("pot " + " ".join(a.replace(" ", "_") for a in arms) + "\n")
        for hi in sorted(data):
            f.write(f"{hi:.4g} " + " ".join(
                f"{data[hi].get(a, float('nan')):.4f}" for a in arms) + "\n")
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="data/test/labels",
                    help="test-set YOLO label directory (source for the box-area CDF)")
    ap.add_argument("--annotations", default="",
                    help="legacy CVAT xml; the pilot export does not match the paper caption")
    ap.add_argument("--curve", default="", help="*_curve.csv from eval_testset (optional)")
    ap.add_argument("--out", default="paper/data")
    ap.add_argument("--npts", type=int, default=200,
                    help="log-spaced sample points for cdf_boxarea.dat")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)

    areas = plant_areas(a.annotations) if a.annotations else areas_from_labels(a.labels_dir)
    cdf = write_cdf(areas, a.out, npts=a.npts)
    if cdf:
        n, s01, s05 = cdf
        print(f"[CDF] {n} plant boxes | sub-T (<0.1%% area)={s01:.1f}%% | <0.5%% area={s05:.1f}%% "
              f"-> data/cdf_boxarea.dat")
        print(f"      caption check: {s01:.1f}%% below 0.1%% area, {s05:.1f}%% below 0.5%% area")
    else:
        print(f"[CDF] could not read {a.annotations}")

    arms = write_recall_curve(a.curve, a.out)
    if arms:
        print(f"[recall-vs-POT] arms={arms} -> data/recall_vs_pot.dat")
    else:
        print("[recall-vs-POT] no *_curve.csv given; pass --curve <file>_curve.csv to generate it")


if __name__ == "__main__":
    main()
