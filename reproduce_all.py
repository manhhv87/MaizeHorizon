#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chay lai MOI phep do dung sau con so trong bai, tu checkpoint den CSV.

  python reproduce_all.py --list                  # xem cac buoc, khong chay gi
  python reproduce_all.py --dry                   # in lenh se chay
  python reproduce_all.py                         # chay het (rat lau)
  python reproduce_all.py --only detection stats  # chi vai nhom
  python reproduce_all.py --skip train counting   # bo vai nhom

`rerun_after_relabel.py` chi phu 13 buoc cua duong phat hien chinh; day la ban day du.
Moi buoc o day deu ghi CSV, va `check_traceability.py` kiem lai rang moi CSV trong
results/ truy duoc ve mot lenh.

## Thu tu phu thuoc

  train     -> checkpoint cho moi thu con lai
  detection -> bang AP/recall theo tang        (uy quyen cho rerun_after_relabel.py)
  scaling   -> quet do phan giai, tuong duong, cu ly khop
  crosssite -> bo 8K: chuyen vung, quet dau vao, cat o, pho cong suat
  rebuttal  -> M2 M3 M4 M4b M9
  counting  -> so cai -> kiem chung -> dem
  remedies  -> track-before-detect
  stats     -> khoang tin cay gop cum, phep kiem dao dau
  figures   -> hinh phu thuoc seed
  checks    -> truy vet + cau truc bai

## Duong dan phai co

Bo 8K nam o o ngoai. Neu khong mount thi nhom `crosssite` tu bo qua chu khong lam
hong ca lo.
"""
import argparse
import os
import subprocess
import sys

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

LAB = "data/test/labels"
IMG = "data/test/images"
IMG_ROOT = "data/images"
HD = "/media/manhhv/DATA/AI/_archive/paper2/test2"
HD1080 = "/media/manhhv/DATA/AI/_archive/paper2/test2_1080"
SEEDS = ["0", "1", "2", "3", "4"]
ARMS = ["stock", "nearonly", "nearfar", "distill", "distill_shuffle", "distill_synth", "nwd"]
CLIP = "IMG_3916"          # clip duy nhat co so cai da kiem chung tay


def steps(seeds):
    S = list(seeds)
    L = ["--labels-dir", LAB, "--images-dir", IMG]
    L2 = ["--labels-dir", f"{HD}/labels", "--images-dir", f"{HD}/images"]
    out = []

    def add(group, name, argv, need=None):
        out.append((group, name, argv, need or []))

    # ---------- 1. huan luyen ----------
    add("train", "missing_seeds",
        ["train_missing_seeds.py", "--seeds", *S, "--workers", "2", "--device", "0",
         "--out", "results/rebuttal/seed_manifest.csv"])

    # ---------- 2. phat hien theo tang ----------
    # rerun_after_relabel.py la nguon su that cho 13 buoc nay; goi lai chu khong chep.
    add("detection", "rerun_after_relabel",
        ["rerun_after_relabel.py", "--seeds", *S])

    add("detection", "cluster_ci",
        ["exp_cluster_ci.py", *L, "--runs", "runs", "--tags", "stock", "nearfar",
         "--seeds", *S, "--iou", "0.3", "--conf", "0.001", "--imgsz", "1280",
         "--max-det", "1000", "--device", "0",
         "--out", "results/detection/cluster_ci.csv",
         "--plants-out", "results/dataset/distinct_plants.csv"])

    add("detection", "ceiling_md1000",
        ["eval_testset.py", *L, "--runs", "runs", "--tags", *ARMS, "--seeds", *S,
         "--iou", "0.3", "--conf", "0.001", "--imgsz", "1280", "--max-det", "1000",
         "--device", "0", "--out", "results/detection/testset_ceiling_md1000.csv"])

    # ---------- 3. quet do phan giai ----------
    add("scaling", "ap_by_native",
        ["exp_ap_by_native.py", *L, "--runs", "runs", "--tag", "nearfar", "--seeds", *S,
         "--imgsz-list", "640", "960", "1280", "1920",
         "--out", "results/scaling/ap_by_native.csv"])

    add("scaling", "equivalence",
        ["exp_equivalence.py", *L, "--runs", "runs", "--tag", "nearfar", "--seeds", *S,
         "--sizes", "640", "960", "1280", "1920",
         "--out", "results/scaling/equivalence.csv"])

    add("scaling", "range_matched",
        ["exp_range_matched.py", *L, "--runs", "runs", "--tags", "stock", "nearfar",
         "--seeds", *S, "--out", "results/scaling/range_matched.csv"])

    add("scaling", "ceiling_sweep",
        ["exp_resolution_sweep.py", *L, "--runs", "runs", "--tag", "nearfar", "--seeds", *S,
         "--imgsz-list", "640", "960", "1280", "1920", "--iou", "0.3", "--conf", "0.001",
         "--device", "0", "--out-prefix", "results/scaling/scaling_ceiling"])

    # ---------- 4. bo thu hai (8K) ----------
    add("crosssite", "E1_zeroshot",
        ["eval_testset.py", *L2, "--runs", "runs", "--tags", "stock", "nearfar",
         "--seeds", *S, "--iou", "0.3", "--conf", "0.001", "--imgsz", "1280",
         "--max-det", "3000", "--device", "0",
         "--out", "results/crosssite/namsach_ceiling.csv"], need=[HD])

    add("crosssite", "E2_input_sweep",
        ["exp_resolution_sweep.py", *L2, "--runs", "runs", "--tag", "stock", "--seeds", *S,
         "--imgsz-list", "1280", "1920", "2560", "3840", "--iou", "0.3", "--conf", "0.001",
         "--max-det", "3000", "--device", "0",
         "--out-prefix", "results/crosssite/hd_sweep"], need=[HD])

    add("crosssite", "E4_sweep_downsampled",
        ["exp_resolution_sweep.py", "--labels-dir", f"{HD1080}/labels",
         "--images-dir", f"{HD1080}/images", "--runs", "runs", "--tag", "stock",
         "--seeds", *S, "--imgsz-list", "1280", "1920", "2560", "3840",
         "--iou", "0.3", "--conf", "0.001", "--max-det", "3000", "--device", "0",
         "--out-prefix", "results/crosssite/hd1080_sweep"], need=[HD1080])

    add("crosssite", "E4_tiling_8k",
        ["exp_sahi_baseline.py", *L2, "--runs", "runs", "--tag", "stock", "--seeds", *S,
         "--tile", "1280", "--imgsz", "1280", "--overlap", "0.2", "--iou", "0.3",
         "--conf", "0.001", "--max-det", "3000", "--device", "0",
         "--out", "results/crosssite/hd_sahi_n5.csv",
         "--per-seed-out", "results/crosssite/hd_sahi_perseed_n5.csv"], need=[HD])

    add("crosssite", "E5_tiling_downsampled",
        ["exp_sahi_baseline.py", "--labels-dir", f"{HD1080}/labels",
         "--images-dir", f"{HD1080}/images", "--runs", "runs", "--tag", "stock",
         "--seeds", *S, "--tile", "320", "--imgsz", "1280", "--overlap", "0.2",
         "--iou", "0.3", "--conf", "0.001", "--max-det", "3000", "--device", "0",
         "--out", "results/crosssite/E5_sahi_8k_ds1080_n5.csv",
         "--per-seed-out", "results/crosssite/E5_sahi_8k_ds1080_perseed_n5.csv"], need=[HD1080])

    add("crosssite", "tiling_1080p",
        ["exp_sahi_baseline.py", *L, "--runs", "runs", "--tag", "nearfar", "--seeds", *S,
         "--tile", "640", "--overlap", "0.2", "--imgsz", "1280", "--iou", "0.3",
         "--conf", "0.001", "--device", "0",
         "--out", "results/baselines/sahi_ap_n5.csv",
         "--per-seed-out", "results/baselines/sahi_perseed_1080p_n5.csv"])

    add("crosssite", "power_spectrum_angular",
        # --mode angular quy tan so goc ve luoi lay mau cua tung cam bien, nen moi
        # thu muc phai kem f/p: NHAN=duong/dan=F_P. Thieu no thi script bao loi.
        ["exp_power_spectrum.py", "--mode", "angular", "--dir", f"1080p={IMG}=1360",
         "--dir", f"8K={HD}/images=5251", "--n-frames", "8", "--crop", "1024",
         "--out", "results/rebuttal/M7_spectrum_angular.csv"], need=[HD])

    add("crosssite", "power_spectrum_nyquist",
        ["exp_power_spectrum.py", "--mode", "nyquist", "--dir", f"1080p={IMG}",
         "--dir", f"8K={HD}/images", "--n-frames", "8", "--crop", "1024",
         "--out", "results/rebuttal/M7_spectrum_nyquist.csv"], need=[HD])

    # ---------- 5. phan bien ----------
    add("rebuttal", "M2_matched_cap",
        ["eval_testset.py", *L, "--runs", "runs", "--tags", "stock", "nearfar",
         "--seeds", *S, "--iou", "0.3", "--conf", "0.001", "--imgsz", "1280",
         "--max-det", "3000", "--device", "0",
         "--out", "results/rebuttal/M2_danphuong_md3000.csv"])

    add("rebuttal", "M3_rtdetr_sweep",
        ["exp_resolution_sweep.py", *L, "--runs", "runs", "--tag", "rtdetr-l",
         "--seeds", *S, "--imgsz-list", "640", "960", "1280", "1920",
         "--iou", "0.3", "--conf", "0.25", "--device", "0",
         "--out-prefix", "results/rebuttal/M3_rtdetr_sweep"])

    add("rebuttal", "M3_h50_by_arch",
        ["exp_h50_by_arch.py", "--prefix", "results/rebuttal/M3_rtdetr_sweep",
         "--label", "rtdetr-l", "--out", "results/rebuttal/M3_h50_by_arch.csv"])

    add("rebuttal", "M4_data_scaling",
        ["exp_data_scaling.py", "--arm", "nearfar", "--fracs", "0.25", "0.5",
         "--seeds", *S, *L, "--imgsz", "1280", "--epochs", "200", "--patience", "20",
         "--batch", "8", "--workers", "2", "--device", "0",
         "--out", "results/rebuttal/M4_data_scaling.csv"])

    add("rebuttal", "M4b_oversample_far",
        ["exp_data_scaling.py", "--arm", "nearfar", "--oversample-far", "20",
         "--seeds", *S, *L, "--imgsz", "1280", "--epochs", "200", "--patience", "20",
         "--batch", "8", "--workers", "2", "--device", "0",
         "--out", "results/rebuttal/M4b_oversample_far.csv"])

    add("rebuttal", "M4_train_scale",
        ["exp_train_scale_dist.py", "--arm", "stock=data/arms/stock/train.txt",
         "--arm", "nearfar=data/arms/nearfar/train.txt", "--test", LAB,
         "--imgsz", "1280", "--out", "results/rebuttal/M4_train_scale.csv"])

    add("rebuttal", "M9_compute_cost",
        ["exp_compute_cost.py", "--images-dir", IMG, "--runs", "runs", "--tag", "nearfar",
         "--seed", "0", "--sizes", "640", "960", "1280", "1920", "2560", "3840",
         "--device", "0", "--out", "results/rebuttal/M9_compute.csv"])

    # ---------- 6. dem cay ----------
    add("counting", "ledger",
        ["furrowmap_ledger.py", "--frames-dir", f"{IMG_ROOT}/{CLIP}_test",
         "--gt-dir", LAB, "--out", f"results/counting/ledger_{CLIP}.json"])

    add("counting", "count",
        ["furrowmap_count.py", "--weights", "runs/nearfar_s0/weights/best.pt",
         "--frames-dir", f"{IMG_ROOT}/{CLIP}_test",
         "--ledger", f"results/counting/ledger_{CLIP}.json",
         "--imgsz", "1280", "--device", "0"])

    # ---------- 7. bien phap ----------
    add("remedies", "track_before_detect",
        ["mve_tbd.py", "--weights", "runs/nearfar_s0/weights/best.pt",
         "--frames-dir", f"{IMG_ROOT}/{CLIP}_test", "--gt-dir", LAB,
         "--imgsz", "1280", "--device", "0"])

    # ---------- 8. thong ke ----------
    add("stats", "tiling_reversal",
        ["exp_seed5_stats.py",
         "--p1080", "results/baselines/sahi_perseed_1080p_n5.csv",
         "--p8k", "results/crosssite/hd_sahi_perseed_n5.csv",
         "--out", "results/rebuttal/tiling_reversal_stats.csv"])

    add("stats", "E5_control",
        ["exp_seed5_stats.py",
         "--p1080", "results/crosssite/E5_sahi_8k_ds1080_perseed_n5.csv",
         "--p8k", "results/crosssite/hd_sahi_perseed_n5.csv",
         "--label-1080", "8K ha mau 1920x1080 (tile 320)",
         "--label-8k", "8K goc (tile 1280)",
         "--out", "results/rebuttal/E5_control_stats.csv"])

    add("stats", "dataset_matrix",
        ["dataset_feature_matrix.py",
         "--out-csv", "results/dataset/_feature_matrix.csv",
         "--out-tex", "paper/sections/_feature_matrix.tex"])

    # ---------- 9. hinh ----------
    for lang, base in (("en", "paper"), ("vi", "paper/vi")):
        extra = [] if lang == "en" else ["--lang", "vi"]
        add("figures", f"fig_scaling_{lang}",
            ["plot_scaling.py", "--prefix", "results/scaling/scaling",
             "--out", f"{base}/figures/fig_scaling", *extra])
        add("figures", f"fig_pr_far_{lang}",
            ["plot_pr_far.py", *L, "--runs", "runs", "--tags", "stock", "nwd", "nearfar", "distill",
             "--seeds", *S, "--iou", "0.3", "--out", f"{base}/figures/fig_pr_far", *extra])
        add("figures", f"fig_cliff_{lang}",
            ["plot_cliff.py", "--csv", "results/detection/testset_ceiling_curve.csv",
             "--out", f"{base}/figures/fig_cliff", *extra])

    # ---------- 10. kiem tra ----------
    add("checks", "traceability", ["check_traceability.py", "--strict"])
    add("checks", "paper_structure", ["check_paper_structure.py"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=SEEDS)
    ap.add_argument("--only", nargs="+", default=None, help="chi cac nhom nay")
    ap.add_argument("--skip", nargs="+", default=[], help="bo qua cac nhom nay")
    ap.add_argument("--list", action="store_true", help="liet ke buoc roi thoat")
    ap.add_argument("--dry", action="store_true", help="in lenh, khong chay")
    ap.add_argument("--keep-going", action="store_true", default=True,
                    help="chay tiep khi mot buoc that bai (mac dinh)")
    a = ap.parse_args()

    plan = steps(a.seeds)
    if a.only:
        plan = [p for p in plan if p[0] in a.only]
    plan = [p for p in plan if p[0] not in a.skip]

    if a.list:
        g = None
        for group, name, argv, need in plan:
            if group != g:
                g = group
                print(f"\n[{group}]")
            miss = [d for d in need if not os.path.exists(d)]
            print(f"  {name:26s} {argv[0]:28s}" + ("  (bo qua: thieu du lieu)" if miss else ""))
        print(f"\n{len(plan)} buoc, {len({p[0] for p in plan})} nhom")
        return

    ok, fail, skipped = [], [], []
    for k, (group, name, argv, need) in enumerate(plan, 1):
        miss = [d for d in need if not os.path.exists(d)]
        if miss:
            print(f"[{k}/{len(plan)}] {group}/{name}: BO QUA, khong thay {miss[0]}")
            skipped.append(f"{group}/{name}")
            continue
        script = os.path.join(HERE, argv[0])
        if not os.path.exists(script):
            print(f"[{k}/{len(plan)}] {group}/{name}: BO QUA, khong co {argv[0]}")
            skipped.append(f"{group}/{name}")
            continue
        print(f"\n########## [{k}/{len(plan)}] {group}/{name}\n$ {' '.join(argv)}", flush=True)
        if a.dry:
            continue
        rc = subprocess.run([PY, script, *argv[1:]], cwd=HERE).returncode
        (ok if rc == 0 else fail).append(f"{group}/{name}")
        if rc != 0:
            print(f"  [!] rc={rc}; chay tiep buoc sau.")

    if not a.dry:
        print(f"\n===== {len(ok)} xong | {len(fail)} loi | {len(skipped)} bo qua =====")
        for f in fail:
            print(f"  [!] {f}")
        for s in skipped:
            print(f"  [-] {s}")


if __name__ == "__main__":
    main()
