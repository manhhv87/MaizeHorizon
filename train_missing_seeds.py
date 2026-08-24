#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the seeds an arm is missing, so every arm in the paper carries the same count.

  python train_missing_seeds.py --seeds 0 1 2 3 4            # train what is missing
  python train_missing_seeds.py --seeds 0 1 2 3 4 --dry      # print the plan only
  python train_missing_seeds.py --seeds 0 1 2 3 4 --only nwd yolo11s

Vi sao can script nay. Moi nhanh co mot entry point rieng (train.py, train_nwd.py,
train_tbxrd_stage2.py, exp_multiarch_train.py, exp_data_scaling.py) voi sieu tham so
rieng. Khi phai nang so seed cho TAT CA cac nhanh de bao cao dong nhat, goi tay tung
lenh mot vua de sai vua khong luu vet. Script nay giu cau hinh cua tung nhanh o mot
cho, chi train seed con THIEU, va ghi mot manifest CSV cho biet da train gi, bao lau.

Cau hinh duoi day duoc doc ra tu runs/<tag>_s0/args.yaml cua chinh cac lan train seed
0-2, nen seed moi khop sieu tham so voi seed cu.

BAY 1 -- duong dan du lieu. args.yaml cua seed 0-2 ghi `datasets/pool_{stock,nearfar}_split`,
thu muc nay KHONG con ton tai (da dep vao _archive). Bo khung tuong duong bay gio nam o
data/arms/<arm>_abs/ voi duong dan TUYET DOI. Da kiem: tap nhan cua pool luu tru trung
khit voi train+valid cua data/arms (3.682 khung nearfar, 118 khung stock), va seed 3/4
train hom 22/08 cho val mAP50 nam trong dai cua seed 0-2 o ca hai nhanh.

BAY 2 -- so worker. 8 worker mac dinh gay deadlock fork giua OpenCV va DataLoader tren
may nay: train dung han sau vai epoch, GPU 0%, khong bao loi gi. Mac dinh o day la 2.
Xem bay #5 trong CLAUDE.md.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

STOCK = "data/arms/stock_abs/data.yaml"      # thay cho datasets/pool_stock_split
NEARFAR = "data/arms/nearfar_abs/data.yaml"  # thay cho datasets/pool_nearfar_split

# tag -> (entry point, argv rieng cua nhanh). {seed} duoc thay luc chay.
# Gia tri lay tu runs/<tag>_s0/args.yaml.
ARMS = {
    # --- YOLOv8-s tren nhanh stock/+Mint o cac kich thuoc dau vao khac nhau
    "stock":     ("train.py", ["--arm", "stock", "--data", STOCK, "--imgsz", "1280",
                               "--epochs", "200", "--patience", "20", "--batch", "8"]),
    "nearfar":   ("train.py", ["--arm", "nearfar", "--data", NEARFAR, "--imgsz", "1280",
                               "--epochs", "200", "--patience", "20", "--batch", "8"]),
    "nf640":     ("train.py", ["--arm", "nearfar", "--name", "nf640", "--data", NEARFAR,
                               "--imgsz", "640", "--epochs", "200", "--patience", "20", "--batch", "8"]),
    "nf960":     ("train.py", ["--arm", "nearfar", "--name", "nf960", "--data", NEARFAR,
                               "--imgsz", "960", "--epochs", "200", "--patience", "20", "--batch", "8"]),
    "nf1920":    ("train.py", ["--arm", "nearfar", "--name", "nf1920", "--data", NEARFAR,
                               "--imgsz", "1920", "--epochs", "200", "--patience", "20", "--batch", "8"]),

    # --- so hang NWD trong loss hoi quy hop (nhanh stock)
    "nwd":       ("train_nwd.py", ["--data", STOCK, "--imgsz", "1280", "--epochs", "200",
                                   "--patience", "20", "--batch", "8", "--name", "nwd_s{seed}",
                                   "--seed", "{seed}"]),

    # --- chung cat cross-resolution + hai doi chung. patience 50, khong phai 20.
    "distill":         ("train_tbxrd_stage2.py", ["--data", NEARFAR, "--imgsz", "1280",
                                                  "--epochs", "200", "--patience", "50", "--batch", "8",
                                                  "--name", "distill_s{seed}", "--seed", "{seed}",
                                                  "--mint-root", "data/mint", "--frames-root", "data/images",
                                                  "--beta", "0.5"]),
    "distill_shuffle": ("train_tbxrd_stage2.py", ["--data", NEARFAR, "--imgsz", "1280",
                                                  "--epochs", "200", "--patience", "50", "--batch", "8",
                                                  "--name", "distill_shuffle_s{seed}", "--seed", "{seed}",
                                                  "--mint-root", "data/mint", "--frames-root", "data/images",
                                                  "--beta", "0.5", "--shuffle"]),
    "distill_synth":   ("train_tbxrd_stage2.py", ["--data", NEARFAR, "--imgsz", "1280",
                                                  "--epochs", "200", "--patience", "50", "--batch", "8",
                                                  "--name", "distill_synth_s{seed}", "--seed", "{seed}",
                                                  "--mint-root", "data/mint", "--frames-root", "data/images",
                                                  "--beta", "0.5", "--synthetic"]),

    # --- cac kien truc khac (nhanh stock). rtdetr-l chay batch 4, khong phai 8.
    "yolo11s":   ("exp_multiarch_train.py", ["--data", STOCK, "--archs", "yolo11s", "--seeds", "{seed}",
                                             "--imgsz", "1280", "--epochs", "200", "--patience", "20",
                                             "--batch", "8"]),
    "yolov10s":  ("exp_multiarch_train.py", ["--data", STOCK, "--archs", "yolov10s", "--seeds", "{seed}",
                                             "--imgsz", "1280", "--epochs", "200", "--patience", "20",
                                             "--batch", "8"]),
    "rtdetr-l":  ("exp_multiarch_train.py", ["--data", STOCK, "--archs", "rtdetr-l", "--seeds", "{seed}",
                                             "--imgsz", "1280", "--epochs", "200", "--patience", "20",
                                             "--batch", "4"]),

    # --- ngan sach giam sat. exp_data_scaling.py tu dung tap con roi goi train.py,
    #     va tu bo qua seed da co checkpoint, nen truyen ca danh sach seed la duoc.
    "scale025":  ("exp_data_scaling.py", ["--arm", "nearfar", "--fracs", "0.25", "--seeds", "{seed}",
                                          "--labels-dir", "data/test/labels", "--images-dir", "data/test/images",
                                          "--imgsz", "1280", "--epochs", "200", "--patience", "20", "--batch", "8"]),
    "scale050":  ("exp_data_scaling.py", ["--arm", "nearfar", "--fracs", "0.5", "--seeds", "{seed}",
                                          "--labels-dir", "data/test/labels", "--images-dir", "data/test/images",
                                          "--imgsz", "1280", "--epochs", "200", "--patience", "20", "--batch", "8"]),
    "farx20":    ("exp_data_scaling.py", ["--arm", "nearfar", "--oversample-far", "20", "--seeds", "{seed}",
                                          "--labels-dir", "data/test/labels", "--images-dir", "data/test/images",
                                          "--imgsz", "1280", "--epochs", "200", "--patience", "20", "--batch", "8"]),
}

# Re truoc, dat truoc: neu phai dung giua chung thi phan da xong van dung duoc.
ORDER = ["nwd", "yolo11s", "yolov10s", "rtdetr-l", "nf640", "scale050", "nf960",
         "farx20", "nf1920", "distill", "distill_shuffle", "distill_synth",
         "stock", "nearfar", "scale025"]


def from_args_yaml(tag, runs, keys=("batch", "imgsz", "patience")):
    """Doc sieu tham so that su cua seed 0. Chep tay la cach de sai nhat.

    Da dinh mot lan: `nf1920` train goc o batch 2 (imgsz 1920 gan day VRAM), nhung
    bang cau hinh chep tay ghi 8. Seed 3 train xong moi phat hien, phai bo di. Batch
    doi ket qua chu khong chi doi toc do, nen mot seed sai batch la mot seed khong so
    duoc voi cac seed kia.
    """
    f = os.path.join(runs, f"{tag}_s0", "args.yaml")
    out = {}
    if not os.path.exists(f):
        return out
    for line in open(f, encoding="utf-8"):
        k, _, v = line.partition(":")
        if k.strip() in keys:
            v = v.strip()
            if v.isdigit():
                out[k.strip()] = v
    return out


def apply_args_yaml(tag, argv, runs):
    """Ghi de cac gia tri trong argv bang gia tri that cua seed 0."""
    real = from_args_yaml(tag, runs)
    changed = []
    for k, v in real.items():
        flag = f"--{k}"
        if flag in argv:
            i = argv.index(flag)
            if argv[i + 1] != v:
                changed.append(f"{k} {argv[i + 1]}->{v}")
                argv[i + 1] = v
    return argv, changed


def done(tag, seed, runs):
    """Mot lan train chi coi la xong khi co CA best.pt LAN results.csv da ghi het.

    best.pt duoc ghi ngay tu epoch 1, nen su ton tai cua no khong chung minh gi.
    Ultralytics ghi results.csv sau moi epoch va ghi dong tong ket cuoi khi ket thuc;
    o day dung tieu chi don gian va an toan: co best.pt, co results.csv, va results.csv
    khong duoc cham vao trong 10 phut vua qua (tuc khong con tien trinh nao dang ghi).
    """
    d = os.path.join(runs, f"{tag}_s{seed}")
    best = os.path.join(d, "weights", "best.pt")
    res = os.path.join(d, "results.csv")
    if not (os.path.exists(best) and os.path.exists(res)):
        return False
    return (time.time() - os.path.getmtime(res)) > 600


def missing(tag, seeds, runs):
    return [s for s in seeds if not done(tag, s, runs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--only", nargs="+", default=None, help="chi cac tag nay")
    ap.add_argument("--skip", nargs="+", default=[], help="bo qua cac tag nay")
    ap.add_argument("--workers", type=int, default=2, help="2 de tranh deadlock; xem docstring")
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default="results/rebuttal/seed_manifest.csv")
    a = ap.parse_args()

    tags = [t for t in ORDER if t in ARMS]
    if a.only:
        tags = [t for t in tags if t in a.only]
    tags = [t for t in tags if t not in a.skip]

    plan = [(t, s) for t in tags for s in missing(t, a.seeds, a.runs)]
    print(f"[i] seeds = {a.seeds} | {len(plan)} lan train con thieu\n")
    for t in tags:
        m = missing(t, a.seeds, a.runs)
        have = [s for s in a.seeds if s not in m]
        print(f"  {t:18s} da co {have}  ->  can train {m if m else '-'}")
    if not plan:
        print("\n[OK] khong thieu gi.")
        return
    if a.dry:
        print("\n--- lenh se chay ---")
        for t, s in plan:
            script, argv = ARMS[t]
            argv = [x.format(seed=s) for x in argv]
            if script == "train.py":
                argv = argv + ["--seeds", str(s)]
            argv, ch = apply_args_yaml(t, argv, a.runs)
            note = f"   [theo args.yaml: {', '.join(ch)}]" if ch else ""
            print(f"  {script} {' '.join(argv)} --workers {a.workers} --device {a.device}{note}")
        return

    rows = [["tag", "seed", "script", "seconds", "returncode", "best_pt"]]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for k, (t, s) in enumerate(plan, 1):
        script, argv = ARMS[t]
        argv = [x.format(seed=s) for x in argv]
        if script == "train.py":
            argv = argv + ["--seeds", str(s)]
        argv, ch = apply_args_yaml(t, argv, a.runs)
        if ch:
            print(f"  [i] {t}: lay theo runs/{t}_s0/args.yaml -> {', '.join(ch)}", flush=True)
        cmd = [PY, os.path.join(HERE, script), *argv,
               "--workers", str(a.workers), "--device", a.device]
        cmd += ["--runs", a.runs]
        print(f"\n########## [{k}/{len(plan)}] {t} seed {s}\n$ {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=HERE).returncode
        dt = time.time() - t0
        best = os.path.join(a.runs, f"{t}_s{s}", "weights", "best.pt")
        rows.append([t, s, script, round(dt, 1), rc, int(os.path.exists(best))])
        print(f"  -> {dt/60:.1f} phut, rc={rc}, best.pt={'co' if os.path.exists(best) else 'KHONG'}")
        with open(a.out, "w", newline="", encoding="utf-8") as f:   # ghi sau moi lan, de mat dien van con
            csv.writer(f).writerows(rows)

    fail = [r for r in rows[1:] if r[4] != 0 or not r[5]]
    print(f"\n[OK] {len(rows)-1} lan train, {len(fail)} that bai -> {a.out}")
    for r in fail:
        print(f"  [!] {r[0]} seed {r[1]}: rc={r[4]} best.pt={'co' if r[5] else 'KHONG'}")


if __name__ == "__main__":
    main()
