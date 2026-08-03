#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrain several architectures to test whether the far-tier floor is specific to YOLOv8-s.

Trains each architecture on the identical arm and seeds, so the only difference is the model.
"""
import argparse
import os
import sys
import warnings

# RT-DETR uses grid_sample, whose backward is not deterministic, so torch warns every epoch
# under deterministic=True. Harmless (warn_only=True); silenced to keep the log readable.
warnings.filterwarnings("ignore", message=r"grid_sampler_2d_backward_cuda does not have a deterministic")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuttal_common import REPO_ROOT  # noqa: E402

# arch -> (kind, COCO-pretrained weights); the .pt downloads on first use
ARCHS = {
    "yolov8s":  ("yolo",   "yolov8s.pt"),
    "yolo11s":  ("yolo",   "yolo11s.pt"),
    "yolov10s": ("yolo",   "yolov10s.pt"),
    "rtdetr-l": ("rtdetr", "rtdetr-l.pt"),
}


def build_model(arch):
    kind, pt = ARCHS[arch]
    if kind == "rtdetr":
        from ultralytics import RTDETR
        return RTDETR(pt)
    from ultralytics import YOLO
    return YOLO(pt)


def _skip_pruner(trainer):
    """This fork calls torch-pruning inside trainer._setup_train with MagnitudePruner
    KHONG DIEU KIEN (chi 'prune'/'sparse_training' moi QUYET DINH co DUNG pruner hay khong).
    torch-pruning KHONG trace duoc RT-DETR (output None) -> crash luc DUNG pruner.
    Voi prune=False & sparse_training=False, pruner khong bao gio duoc dung -> dat sentinel
    non-None de BO QUA buoc dung (tranh trace RT-DETR). Callback chay o 'on_pretrain_routine_start'
    (before the trainer builds the pruner)."""
    trainer.pruner = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YAML per-plant (train/val/test + nc + names). "
                                                  "Dung DUNG data yaml cua arm stock/+Mint de so sanh.")
    ap.add_argument("--archs", nargs="+", default=list(ARCHS),
                    help=f"one of: {list(ARCHS)}")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="0")
    ap.add_argument("--optimizer", default="auto",
                    help="auto (khuyen nghi) | SGD | AdamW. auto: YOLO->SGD, RT-DETR->AdamW.")
    ap.add_argument("--lr0", type=float, default=None, help="leave unset to use each architecture default")
    ap.add_argument("--runs", default=os.path.join(REPO_ROOT, "runs"))
    ap.add_argument("--resume-tag", default=None)
    ap.add_argument("--resume-seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan without training")
    a = ap.parse_args()

    for arch in a.archs:
        if arch not in ARCHS:
            raise SystemExit(f"unsupported arch '{arch}'; choose from {list(ARCHS)}")

    plan = [(arch, s) for arch in a.archs for s in a.seeds]
    if a.resume_tag is not None:
        plan = [(a.resume_tag, a.resume_seed)]
    print(f"[i] {len(plan)} run | imgsz={a.imgsz} epochs={a.epochs} batch={a.batch} "
          f"optimizer={a.optimizer} | runs={a.runs}")
    for arch, seed in plan:
        print(f"    - {arch}_s{seed}")
    if a.dry_run:
        return

    for arch, seed in plan:
        name = f"{arch}_s{seed}"
        out = os.path.join(a.runs, name)
        last = os.path.join(out, "weights", "last.pt")
        resume = a.resume_tag is not None and os.path.exists(last)
        print(f"\n========== TRAIN {name} ==========")
        train_kwargs = dict(
            data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            device=a.device, seed=seed, deterministic=True, patience=a.patience,
            optimizer=a.optimizer, project=a.runs, name=name, exist_ok=True,
            prune=False,   # fork tich hop torch-pruning; PHAI tat (giong train.py cua repo)
        )
        if a.lr0 is not None:
            train_kwargs["lr0"] = a.lr0
        is_rtdetr = ARCHS[arch][0] == "rtdetr"
        if resume:
            model = build_model(arch).__class__(last)
            if is_rtdetr:
                model.add_callback("on_pretrain_routine_start", _skip_pruner)
            model.train(resume=True)
        else:
            model = build_model(arch)
            if is_rtdetr:                       # bo qua buoc dung pruner (RT-DETR khong trace duoc)
                model.add_callback("on_pretrain_routine_start", _skip_pruner)
            model.train(**train_kwargs)
        # no model.val() here: the real evaluation is POT-stratified and ignore-aware on the
        # hand-labelled test clips (exp_multiarch_eval.py), not mAP on the data.yaml split
        best = os.path.join(a.runs, name, "weights", "best.pt")
        if os.path.exists(best):
            print(f"[{name}] DONE -> {best}")
        else:
            print(f"[{name}] !! no best.pt at {best}; training may have failed, see the log above.")
    print("\n-> evaluate with exp_multiarch_eval.py on the hand-labelled test clips:")
    print("   python exp_multiarch_eval.py --labels-dir <LAB> --images-dir <IMG> \\")
    print(f"       --runs {a.runs} --tags stock " + " ".join(a.archs)
          + " --seeds " + " ".join(map(str, a.seeds)) + " --imgsz 1280 --iou 0.3 --device 0")


if __name__ == "__main__":
    main()
