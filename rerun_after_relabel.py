#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run every evaluation that depends on the test labels.

No retraining: the new labels are test labels, so the checkpoints stay as they are. Each step calls
the existing script with the paper protocol (POT-stratified, ignore-aware, one-to-one greedy IoU),
so the numbers remain directly comparable with the tables.

  python rerun_after_relabel.py                          # everything
  python rerun_after_relabel.py --only multiarch,contrast
  python rerun_after_relabel.py --dry                    # print commands only
"""
import argparse
import glob
import os
import subprocess
import sys

# ===================== CONFIG (edit for your machine) =====================
LABELS_DIR = r"data/test/labels"   # thu muc *.txt TEST (DA them far moi)
IMAGES_DIR = r"data/images"                # root anh test (tim de quy theo basename)
RUNS       = "runs"                                         # <tag>_s<seed>/weights/best.pt
DEVICE     = "0"                                            # GPU id, hoac "cpu"
SEEDS      = [0, 1, 2]   # mac dinh; --seeds ghi de. stock/nearfar nay co them 3, 4.
                         # Cac script eval deu loc theo checkpoint co that va ghi n_seeds
                         # tung dong, nen truyen 0..4 cho ca lo la an toan: nhanh nao chi
                         # co 3 seed thi van dung 3.
IMGSZ      = 1280

ARMS   = ["stock", "nearonly", "nearfar", "distill", "distill_shuffle", "distill_synth", "nwd"]  # eval_testset/eval_ap (nwd cho sec.res-scaling)
# the YOLOv8-s row of tab:multiarch uses the 'stock' tag; there is usually no runs/yolov8s_s*
ARCHS  = ["stock", "yolo11s", "yolov10s", "rtdetr-l"]       # exp_multiarch_eval (san far bat bien theo arch)
TRAINMATCHED = [("nf640", 640), ("nf960", 960), ("nearfar", 1280), ("nf1920", 1920)]  # (tag,imgsz) da train -> chi EVAL lai (Table trainmatched)
SWEEP_TAG = "nearfar"                                       # arm cho resolution sweep + SAHI + contrast (=+Mint)
BURST = None   # tuy chon burst-SR theo clip: (frames_dir, gt_dir); None = bo qua.
               # vd: (r"dataset/corn_plant/to_label/IMG_3916", LABELS_DIR)
# ========================================================================

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def weights_csv(tag):
    """Existing best.pt paths as a comma-separated string, for scripts taking --weights."""
    ws = [os.path.join(RUNS, f"{tag}_s{s}", "weights", "best.pt") for s in SEEDS]
    return ",".join(w for w in ws if os.path.exists(w))


def far_n_scan():
    """Count far GT (POT < 16) in the test labels; purely geometric, no model needed."""
    try:
        import cv2  # noqa: F401
        from eval_testset import build_image_index, read_gt, stratum
    except Exception as e:  # noqa: BLE001
        print(f"[far-N scan skipped: {e}]")
        return
    import cv2
    idx = build_image_index(IMAGES_DIR)
    tiers = {"near": 0, "mid": 0, "far": 0}
    nimg = 0
    for lp in sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = idx.get(stem)
        if not ip:
            continue
        im = cv2.imread(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        nimg += 1
        scale = IMGSZ / max(w, h)
        plants, _ = read_gt(lp, w, h)
        for b in plants:
            tiers[stratum(b, "pot", scale)] += 1
    tot = sum(tiers.values())
    print("\n================= FAR-N sau relabel =================")
    print(f"  {nimg} test images | plant GT = {tot}  "
          f"(near {tiers['near']} / mid {tiers['mid']} / FAR {tiers['far']})")
    print(f"  -> update the far-tier count to {tiers['far']} in: abstract, methods, "
          f"caption tab:detection/tab:cliff/tab:multiarch, va Wilson CI.")
    print("=====================================================\n")


def build_steps():
    """(name, argv) per step; argv[0] is the script name."""
    L = ["--labels-dir", LABELS_DIR, "--images-dir", IMAGES_DIR]
    base = ["--runs", RUNS, "--seeds", *map(str, SEEDS), "--device", DEVICE, "--imgsz", str(IMGSZ)]
    steps = []
    # 1) per-tier detection @conf van hanh (tab:detection/tab:tier) -- IoU 0.3 (chinh) & 0.5
    steps.append(("detection_iou03", ["eval_testset.py", *L, *base, "--tags", *ARMS,
                                       "--iou", "0.3", "--conf", "0.25", "--out", "results/detection/testset_iou03.csv"]))
    steps.append(("detection_iou05", ["eval_testset.py", *L, *base, "--tags", *ARMS,
                                       "--iou", "0.5", "--conf", "0.25", "--out", "results/detection/testset_iou05.csv"]))
    # 2) far recall at the ceiling (conf->0) per arm (tab:cliff)
    steps.append(("cliff_ceiling", ["eval_testset.py", *L, *base, "--tags", *ARMS,
                                     "--iou", "0.3", "--conf", "0.001", "--out", "results/detection/testset_ceiling.csv"]))
    # 2b) per-tier AP (integrates PR, independent of conf) for tab:detection, NWD and SAHI
    steps.append(("ap_iou03", ["eval_ap.py", *L, *base, "--tags", *ARMS, "--iou", "0.3",
                               "--out", "results/detection/testset_ap03.csv"]))
    steps.append(("ap_iou05", ["eval_ap.py", *L, *base, "--tags", *ARMS, "--iou", "0.5",
                               "--out", "results/detection/testset_ap05.csv"]))
    # 3) cross-architecture floor with Wilson CI at the true n (tab:multiarch)
    steps.append(("multiarch", ["exp_multiarch_eval.py", *L, "--runs", RUNS,
                                "--seeds", *map(str, SEEDS), "--tags", *ARCHS,
                                "--imgsz", str(IMGSZ), "--iou", "0.3", "--device", DEVICE,
                                "--out", "results/baselines/rebuttal_multiarch.csv"]))
    # 4) resolution sweep (fig scaling / bin far & small physical)
    steps.append(("resolution", ["exp_resolution_sweep.py", *L, "--runs", RUNS, "--tag", SWEEP_TAG,
                                 "--seeds", *map(str, SEEDS), "--imgsz-list", "640", "960", "1280", "1920",
                                 "--device", DEVICE, "--out-prefix", "results/scaling/scaling"]))
    # 4b) train-matched sweep: each (tag, imgsz) is already trained, so only re-evaluate
    for tm_tag, tm_sz in TRAINMATCHED:
        steps.append((f"trainmatched_{tm_sz}",
                      ["exp_resolution_sweep.py", *L, "--runs", RUNS, "--tag", tm_tag,
                       "--seeds", *map(str, SEEDS), "--imgsz-list", str(tm_sz), "--iou", "0.3",
                       "--conf", "0.25", "--device", DEVICE, "--out-prefix", f"results/scaling/scaling_tm_{tm_sz}"]))
    # 5) contrast terciles (h50 vs ExG)
    wc = weights_csv(SWEEP_TAG)
    if wc:
        steps.append(("contrast", ["exp_contrast.py", *L, "--weights", wc, "--imgsz", str(IMGSZ),
                                   "--device", DEVICE, "--out", "results/baselines/rebuttal_contrast"]))
    else:
        print(f"[contrast skipped: no best.pt for '{SWEEP_TAG}' under {RUNS}]")
    # 6) SAHI baseline (far AP tiled vs full-frame)
    steps.append(("sahi", ["exp_sahi_baseline.py", *L, "--runs", RUNS, "--tag", SWEEP_TAG,
                          "--seeds", *map(str, SEEDS), "--imgsz", str(IMGSZ), "--device", DEVICE,
                          "--out", "results/baselines/sahi_ap.csv"]))
    # 7) burst SR (optional, per clip)
    if BURST:
        wb = weights_csv(SWEEP_TAG)
        frames, gt = BURST
        if wb:
            steps.append(("burst_sr", ["exp_burst_sr.py", "--weights", wb, "--frames-dir", frames,
                                      "--gt-dir", gt, "--imgsz", str(IMGSZ), "--device", DEVICE,
                                      "--out", "results/baselines/rebuttal_burst_sr.csv"]))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only these steps, e.g. multiarch,contrast")
    ap.add_argument("--dry", action="store_true", help="print the commands without running them")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="ghi de SEEDS o dau file, vd --seeds 0 1 2 3 4")
    a = ap.parse_args()
    if a.seeds:
        global SEEDS
        SEEDS = a.seeds
        print(f"[i] seeds = {SEEDS}")
    want = {x.strip() for x in a.only.split(",")} if a.only else None

    if not os.path.isdir(LABELS_DIR):
        raise SystemExit(f"LABELS_DIR does not exist: {LABELS_DIR} (edit CONFIG at the top).")
    far_n_scan()

    ok, fail, skip = [], [], []
    for name, argv in build_steps():
        if want and not any(name.startswith(w) for w in want):
            skip.append(name)
            continue
        script = os.path.join(HERE, argv[0])
        if not os.path.exists(script):
            print(f"[skip {name}: {argv[0]} not found]")
            skip.append(name)
            continue
        cmd = [PY, script, *argv[1:]]
        print(f"\n########## [{name}]\n$ {' '.join(argv)}")
        if a.dry:
            continue
        rc = subprocess.run(cmd, cwd=HERE).returncode
        (ok if rc == 0 else fail).append(name)
        if rc != 0:
            print(f"  [!] '{name}' failed (rc={rc}); continuing with the next step.")

    if not a.dry:
        print(f"\n=== XONG.  OK={ok}  LOI={fail}  BO_QUA={skip} ===")
        print("New CSVs are in results/. Update the paper:")
        print("  - tab:detection / tab:cliff : cot 'far' (testset_iou03/05.csv, results/detection/testset_ceiling.csv)")
        print("  - tab:multiarch            : recall + Wilson CI (rebuttal_multiarch_far_ceiling.csv) -- CI da o n THAT")
        print("  - fig scaling / trainmatched: bin far & 24-48px (scaling_*.csv)")
        print("  - contrast h50             : results/baselines/rebuttal_contrast*")
        print("  - replace the old far-tier count in the abstract, methods and captions")


if __name__ == "__main__":
    main()
