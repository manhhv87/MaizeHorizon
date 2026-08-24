#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Duong cong quy mo du lieu: recall tang xa co bao hoa theo so khung huan luyen khong?

Phan bien M4 hoi mot cau ma bai chua tra loi duoc: neu tap huan luyen gan nhu khong co
muc tieu tang xa (do duoc: 0.1% so voi 2.0% o tap test), thi ket luan "tang input khong
mo rong duoc tam" noi ve GIAM SAT chu khong noi ve CAM BIEN.

Phep thu: huan luyen lai cung mot kien truc tren 25%, 50%, 100% so khung, roi do recall
tang xa tren cung tap test.

  - Neu recall tang xa BAO HOA truoc 100%  -> them du lieu khong cuu duoc tang xa,
    nen gioi han khong den tu luong giam sat, va lap luan cua bai vung hon.
  - Neu van con DOC LEN o 100%             -> gioi han (mot phan) den tu giam sat,
    va ket luan phai viet lai.

Cac tap con lay ngau nhien co seed tu train.txt cua mot nhanh. Tap test va tap valid
khong doi, nen moi diem tren duong cong so sanh duoc voi nhau.

  "$PY" exp_data_scaling.py --arm nearfar --fracs 0.25 0.5 --seed 0 \\
      --labels-dir data/test/labels --images-dir data/test/images \\
      --out results/rebuttal/M4_data_scaling.csv
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
import sys

PY = sys.executable


def far_frames(arm_dir, imgsz=1280, W=1920, H=1080, thresh=16.0):
    """Cac dong trong train.txt ma khung do co it nhat mot box duoi nguong POT."""
    from eval_testset import read_gt
    sc = imgsz / max(W, H)
    out = []
    for ln in open(os.path.join(arm_dir, "train.txt"), encoding="utf-8"):
        rel = ln.strip()
        if not rel:
            continue
        lp = os.path.join("data", rel.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
        if not os.path.exists(lp):
            continue
        if any((b[3] - b[1]) * sc < thresh for b in read_gt(lp, W, H, 0, 1)[0]):
            out.append(rel)
    return out


def build_oversampled(arm_dir, repeat, seed, out_dir):
    """Doi THANH PHAN ma giu nguyen TONG SO khung, de so sanh khop chi phi.

    Cac khung chua box tang xa duoc lap `repeat` lan; phan con lai lay ngau nhien
    cho du tong so bang tap goc. Model vi the thay vi du tang xa nhieu gap `repeat`
    lan ma khong ton them mot buoc huan luyen nao.
    """
    allf = [l.strip() for l in open(os.path.join(arm_dir, "train.txt"), encoding="utf-8") if l.strip()]
    far = far_frames(arm_dir)
    rest = [x for x in allf if x not in set(far)]
    slots = far * repeat
    rng = random.Random(seed)
    need = len(allf) - len(slots)
    if need < 0:
        slots = slots[:len(allf)]; need = 0
    keep = slots + rng.sample(rest, need)
    rng.shuffle(keep)
    return keep, far, allf


def write_arm(arm_dir, keep, tag, out_dir):
    """Ghi mot nhanh: train.txt/valid.txt duong dan TUYET DOI, symlink anh, chep nhan can dung."""
    os.makedirs(out_dir, exist_ok=True)
    src_tag = os.path.basename(arm_dir)
    data_root = os.path.abspath(os.path.join(out_dir, "..", ".."))

    def absify(rel):
        return os.path.join(data_root, rel.replace(f"arms/{src_tag}/", f"arms/{tag}/"))

    open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8").write(
        "\n".join(absify(p) for p in keep) + "\n")
    vlines = [l.strip() for l in open(os.path.join(arm_dir, "valid.txt"), encoding="utf-8") if l.strip()]
    open(os.path.join(out_dir, "valid.txt"), "w", encoding="utf-8").write(
        "\n".join(absify(p) for p in vlines) + "\n")

    img_dst = os.path.join(out_dir, "images")
    if not os.path.lexists(img_dst):
        os.symlink(os.path.relpath(os.path.realpath(os.path.join(arm_dir, "images")), out_dir), img_dst)
    lab_dst = os.path.join(out_dir, "labels")
    if os.path.islink(lab_dst):
        os.unlink(lab_dst)
    src_lab = os.path.realpath(os.path.join(arm_dir, "labels"))
    for rel in set(list(keep) + vlines):
        r = rel.split(f"arms/{src_tag}/images/", 1)[-1].rsplit(".", 1)[0] + ".txt"
        s_, d_ = os.path.join(src_lab, r), os.path.join(lab_dst, r)
        os.makedirs(os.path.dirname(d_), exist_ok=True)
        if os.path.exists(s_) and not os.path.exists(d_):
            shutil.copy(s_, d_)
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {data_root}\ntrain: arms/{tag}/train.txt\nval: arms/{tag}/valid.txt\n\n"
                f"names:\n  0: plant\n  1: ignore\n")


def build_subset(arm_dir, frac, seed, out_dir):
    """Lay mau ngau nhien co seed tu train.txt cua mot nhanh."""
    lines = [l.strip() for l in open(os.path.join(arm_dir, "train.txt"), encoding="utf-8") if l.strip()]
    keep = sorted(random.Random(seed).sample(lines, max(1, int(round(len(lines) * frac)))))
    write_arm(arm_dir, keep, os.path.basename(out_dir), out_dir)
    return len(keep), len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="nearfar")
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.25, 0.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="moi seed rut mot tap con RIENG va mot khoi tao rieng, nen"
                         " do tan mac bao gom ca bien thien do rut mau")
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8,
                    help="chuyen thang cho train.py; xem ghi chu deadlock o do")
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out")
    ap.add_argument("--skip-train", action="store_true", help="chi danh gia lai checkpoint da co")
    ap.add_argument("--oversample-far", type=int, default=0,
                    help="thay vi quet ti le, dung MOT tap cung kich thuoc nhung lap cac khung\n"
                         "co box tang xa N lan (doi thanh phan, giu nguyen chi phi)")
    a = ap.parse_args()

    arm_dir = os.path.join("data", "arms", a.arm)
    tags = []

    if a.oversample_far:
        tag = f"farx{a.oversample_far}"
        out_dir = os.path.join("data", "arms", tag)
        for sd in a.seeds:
            keep, far, allf = build_oversampled(arm_dir, a.oversample_far, sd, out_dir)
            write_arm(arm_dir, keep, tag, out_dir)
            print(f"[{tag} seed {sd}] {len(keep)} suat = {len(far)} khung co tang xa x{a.oversample_far}"
                  f" + {len(keep)-len(far)*a.oversample_far} khung khac (goc {len(allf)})", flush=True)
            if os.path.exists(os.path.join(a.runs, f"{tag}_s{sd}", "weights", "best.pt")):
                print("  da co checkpoint, bo qua train", flush=True); continue
            if not a.skip_train:
                subprocess.run([PY, "train.py", "--arm", a.arm, "--name", tag,
                                "--data", os.path.join(out_dir, "data.yaml"),
                                "--seeds", str(sd), "--imgsz", str(a.imgsz),
                                "--epochs", str(a.epochs), "--patience", str(a.patience),
                                "--batch", str(a.batch), "--workers", str(a.workers), "--device", a.device,
                                "--runs", a.runs], check=True)
        tags.append((tag, len(keep), float(a.oversample_far)))
        a.fracs = []

    for fr in a.fracs:
        tag = f"scale{int(round(fr*100)):03d}"
        for sd in a.seeds:
            # tap con dung lai theo seed ngay truoc khi train seed do, nen moi seed
            # thay mot bo khung khac; deterministic vi random.Random(sd)
            n_, tot = build_subset(arm_dir, fr, sd, os.path.join("data", "arms", tag))
            print(f"[{tag} seed {sd}] {n_}/{tot} khung ({100*n_/tot:.0f}%)", flush=True)
            if os.path.exists(os.path.join(a.runs, f"{tag}_s{sd}", "weights", "best.pt")):
                print(f"  da co checkpoint, bo qua train", flush=True); continue
            if not a.skip_train:
                subprocess.run([PY, "train.py", "--arm", a.arm, "--name", tag,
                                "--data", os.path.join(arm_dir.replace(a.arm, tag), "data.yaml"),
                                "--seeds", str(sd), "--imgsz", str(a.imgsz),
                                "--epochs", str(a.epochs), "--patience", str(a.patience),
                                "--batch", str(a.batch), "--workers", str(a.workers), "--device", a.device,
                                "--runs", a.runs], check=True)
        tags.append((tag, n_, fr))

    # diem 100% dung lai checkpoint san co cua chinh nhanh do
    tags.append((a.arm, sum(1 for _ in open(os.path.join(arm_dir, "train.txt"))), 1.0))

    out_csv = a.out or "M4_data_scaling.csv"
    tmp = out_csv.replace(".csv", "_eval.csv")
    subprocess.run([PY, "eval_testset.py", "--labels-dir", a.labels_dir,
                    "--images-dir", a.images_dir, "--runs", a.runs,
                    "--tags"] + [t for t, _, _ in tags] +
                   ["--seeds"] + [str(x) for x in a.seeds] +
                   ["--iou", "0.3", "--conf", "0.001",
                    "--imgsz", str(a.imgsz), "--max-det", "1000",
                    "--device", a.device, "--out", tmp], check=True)

    rd = list(csv.DictReader(open(tmp)))
    rec = {(r["model"], r["stratum"]): r["recall_mean"] for r in rd}
    sd_ = {(r["model"], r["stratum"]): r.get("recall_std", "") for r in rd}
    rows = []
    print(f"\n{'nhanh':10} {'khung':>6} {'phan':>5}  {'near':>7} {'mid':>7} {'far':>7}")
    for tag, n, fr in tags:
        v = [rec.get((tag, s), "") for s in ("near", "mid", "far")]
        e = [sd_.get((tag, s), "") for s in ("near", "mid", "far")]
        print(f"{tag:10} {n:>6} {fr:>5.2f}  " + " ".join(
            f"{float(x):.4f}+/-{float(y or 0):.4f}" if x else f"{'-':>13}" for x, y in zip(v, e)))
        rows.append([tag, n, f"{fr:.2f}"] + [q for pair in zip(v, e) for q in pair])
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "n_train_frames", "fraction", "recall_near", "std_near",
                        "recall_mid", "std_mid", "recall_far", "std_far"])
            w.writerows(rows)
        print(f"\n[OK] -> {a.out}")


if __name__ == "__main__":
    main()
