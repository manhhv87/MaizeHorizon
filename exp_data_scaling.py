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


def build_subset(arm_dir, frac, seed, out_dir):
    """Tao mot nhanh con: train.txt lay mau, valid.txt giu nguyen, symlink dung chung anh."""
    os.makedirs(out_dir, exist_ok=True)
    lines = [l.strip() for l in open(os.path.join(arm_dir, "train.txt"), encoding="utf-8") if l.strip()]
    rng = random.Random(seed)
    keep = rng.sample(lines, max(1, int(round(len(lines) * frac))))
    keep.sort()
    tag = os.path.basename(out_dir)
    # duong dan trong train.txt tuong doi so voi 'path:' cua data.yaml -> phai doi ten nhanh
    src_tag = os.path.basename(arm_dir)
    # Duong dan TUYET DOI. Ultralytics chi viet lai duong dan bat dau bang './'; con lai
    # no giu nguyen va giai theo thu muc lam viec, nen duong dan tuong doi trong train.txt
    # chi chay khi DATASETS_DIR duoc tro dung. Ghi tuyet doi thi thi nghiem tu dung duoc.
    data_root = os.path.abspath(os.path.join(out_dir, "..", ".."))
    def absify(rel):
        return os.path.join(data_root, rel.replace(f"arms/{src_tag}/", f"arms/{tag}/"))
    open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8").write(
        "\n".join(absify(p) for p in keep) + "\n")
    vlines = [l.strip() for l in open(os.path.join(arm_dir, "valid.txt"), encoding="utf-8") if l.strip()]
    open(os.path.join(out_dir, "valid.txt"), "w", encoding="utf-8").write(
        "\n".join(absify(p) for p in vlines) + "\n")
    # anh: symlink (nhieu GB, khong duoc chep). nhan: CHEP that, chi nhung file can dung.
    # Ultralytics ghi file .cache canh thu muc nhan; neu nhan la symlink dung chung thi
    # cac tap con dam cache cua nhau va scan ra 0 anh hop le.
    img_dst = os.path.join(out_dir, "images")
    if not os.path.lexists(img_dst):
        os.symlink(os.path.relpath(os.path.realpath(os.path.join(arm_dir, "images")), out_dir), img_dst)
    lab_dst = os.path.join(out_dir, "labels")
    if os.path.islink(lab_dst):
        os.unlink(lab_dst)
    src_lab = os.path.realpath(os.path.join(arm_dir, "labels"))
    for rel in list(keep) + vlines:
        r = rel.split(f"arms/{src_tag}/images/", 1)[-1].rsplit(".", 1)[0] + ".txt"
        s_, d_ = os.path.join(src_lab, r), os.path.join(lab_dst, r)
        os.makedirs(os.path.dirname(d_), exist_ok=True)
        if os.path.exists(s_) and not os.path.exists(d_):
            shutil.copy(s_, d_)
    # 'path' phai TUYET DOI: Ultralytics giai duong dan tuong doi theo DATASETS_DIR cua no,
    # khong theo vi tri file yaml (chinh data.yaml goc cua repo co ghi chu canh bao dieu nay).
    abs_data = os.path.abspath(os.path.join(out_dir, "..", ".."))
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {abs_data}\ntrain: arms/{tag}/train.txt\nval: arms/{tag}/valid.txt\n\n"
                f"names:\n  0: plant\n  1: ignore\n")
    return len(keep), len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="nearfar")
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.25, 0.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out")
    ap.add_argument("--skip-train", action="store_true", help="chi danh gia lai checkpoint da co")
    a = ap.parse_args()

    arm_dir = os.path.join("data", "arms", a.arm)
    tags = []
    for fr in a.fracs:
        tag = f"scale{int(round(fr*100)):03d}"
        n, tot = build_subset(arm_dir, fr, a.seed, os.path.join("data", "arms", tag))
        print(f"[{tag}] {n}/{tot} khung ({100*n/tot:.0f}%)")
        tags.append((tag, n, fr))
        if not a.skip_train:
            subprocess.run([PY, "train.py", "--arm", a.arm, "--name", tag,
                            "--data", os.path.join(arm_dir.replace(a.arm, tag), "data.yaml"),
                            "--seeds", str(a.seed), "--imgsz", str(a.imgsz),
                            "--epochs", str(a.epochs), "--patience", str(a.patience),
                            "--batch", str(a.batch), "--device", a.device,
                            "--runs", a.runs], check=True)

    # diem 100% dung lai checkpoint san co cua chinh nhanh do
    tags.append((a.arm, sum(1 for _ in open(os.path.join(arm_dir, "train.txt"))), 1.0))

    out_csv = a.out or "M4_data_scaling.csv"
    tmp = out_csv.replace(".csv", "_eval.csv")
    subprocess.run([PY, "eval_testset.py", "--labels-dir", a.labels_dir,
                    "--images-dir", a.images_dir, "--runs", a.runs,
                    "--tags"] + [t for t, _, _ in tags] +
                   ["--seeds", str(a.seed), "--iou", "0.3", "--conf", "0.001",
                    "--imgsz", str(a.imgsz), "--max-det", "1000",
                    "--device", a.device, "--out", tmp], check=True)

    rec = {(r["model"], r["stratum"]): r["recall_mean"] for r in csv.DictReader(open(tmp))}
    rows = []
    print(f"\n{'nhanh':10} {'khung':>6} {'phan':>5}  {'near':>7} {'mid':>7} {'far':>7}")
    for tag, n, fr in tags:
        v = [rec.get((tag, s), "") for s in ("near", "mid", "far")]
        print(f"{tag:10} {n:>6} {fr:>5.2f}  " + " ".join(f"{float(x):7.4f}" if x else f"{'-':>7}" for x in v))
        rows.append([tag, n, f"{fr:.2f}"] + list(v))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arm", "n_train_frames", "fraction", "recall_near", "recall_mid", "recall_far"])
            w.writerows(rows)
        print(f"\n[OK] -> {a.out}")


if __name__ == "__main__":
    main()
