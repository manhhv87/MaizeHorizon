#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate multi-seed test results and emit a LaTeX table row.

For each tag, evaluates every seed (best.pt), computes mean +/- std for P/R/F1/mAP and
params/GFLOPs, prints a summary table and a ready-to-paste LaTeX row, and runs a seed-paired
t-test against the baseline for one metric.
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

METS = ["P", "R", "F1", "map50", "map"]   # cot detection cua tabMain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", default=None, help="runs directory (default: <repo>/runs)")
    ap.add_argument("--tags", nargs="+", default=["baseline", "p345_full", "p45_full"],
                    help="run prefix; each tag has <tag>_s{seed}")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="display names for the LaTeX row (must match --tags); defaults to the tag")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--baseline", default="baseline", help="baseline tag for the paired t-test")
    ap.add_argument("--metric", default="map50", choices=METS, help="metric for the paired t-test")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    import numpy as np
    import ultralytics.nn.tasks as tasks
    from ultralytics import YOLO
    try:
        from ultralytics.utils.sgc_ext import SGCDetectionLoss
        tasks.DetectionModel.init_criterion = lambda self: SGCDetectionLoss(self)
    except Exception:
        pass

    project = a.project or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    labels = a.labels if (a.labels and len(a.labels) == len(a.tags)) else a.tags

    def metrics_one(w):
        """1 checkpoint -> dict {P,R,F1,map50,map,params,gflops}."""
        m = YOLO(w)
        info = m.info(verbose=False)                  # (layers, params, gradients, GFLOPs)
        params = float(info[1]) / 1e6
        gflops = float(info[3])
        b = m.val(data=a.data, split="test", device=a.device, verbose=False).box
        P, R, m50, m95 = float(b.mp), float(b.mr), float(b.map50), float(b.map)
        F1 = 2 * P * R / (P + R + 1e-9)
        return {"P": P, "R": R, "F1": F1, "map50": m50, "map": m95,
                "params": params, "gflops": gflops}

    res = {}   # tag -> {seed: metrics_dict}
    for tag in a.tags:
        d = {}
        for s in a.seeds:
            w = os.path.join(project, f"{tag}_s{s}", "weights", "best.pt")
            if not os.path.exists(w):
                print(f"  [skip] missing {w}")
                continue
            d[s] = metrics_one(w)
        res[tag] = d

    def agg(tag, key):
        d = res[tag]
        v = np.array([d[s][key] for s in sorted(d)])
        return (float(v.mean()), float(v.std()), len(v)) if len(v) else (float("nan"), 0.0, 0)

    # ---- summary table ----
    print("\n== TEST (mean +/- std over seeds) ==")
    print(f"{'tag':18s} {'P':>13} {'R':>13} {'F1':>13} {'mAP50':>13} {'mAP50-95':>13} {'Par(M)':>7} {'GFLOPs':>7}")
    for tag in a.tags:
        if not res[tag]:
            continue
        cells = []
        for k in METS:
            mu, sd, n = agg(tag, k)
            cells.append(f"{mu:.3f}+/-{sd:.3f}")
        pa, _, _ = agg(tag, "params"); gf, _, _ = agg(tag, "gflops")
        print(f"{tag:18s} " + " ".join(f"{c:>13}" for c in cells) + f" {pa:7.2f} {gf:7.1f}")

    # ---- LaTeX row ----
    print("\n% ==== COPY cac dong duoi vao tabMain (thay [XX]) ====")
    for tag, lab in zip(a.tags, labels):
        if not res[tag]:
            continue
        c = {k: agg(tag, k) for k in METS}
        pa, _, _ = agg(tag, "params"); gf, _, _ = agg(tag, "gflops")
        row = " & ".join(f"{c[k][0]:.3f}$\\pm${c[k][1]:.3f}" for k in METS)
        print(f"{lab} & {row} & {pa:.1f} & {gf:.1f} \\\\")

    # ---- paired t-test ----
    try:
        from scipy import stats
        b = res.get(a.baseline)
        if b:
            print(f"\n== paired t-test on {a.metric} vs {a.baseline} (paired by seed) ==")
            for tag in a.tags:
                if tag == a.baseline or not res[tag]:
                    continue
                common = sorted(set(res[tag]) & set(b))
                if len(common) < 2:
                    print(f"{tag:18s} (chi {len(common)} seed chung)")
                    continue
                vt = np.array([res[tag][s][a.metric] for s in common])
                vb = np.array([b[s][a.metric] for s in common])
                t, p = stats.ttest_rel(vt, vb)
                print(f"{tag:18s} delta={vt.mean()-vb.mean():+.4f}  p={p:.4f}  (n={len(common)})")
    except ImportError:
        print("(install scipy for a p-value: pip install scipy)")


if __name__ == "__main__":
    main()
