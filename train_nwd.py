#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train YOLOv8-s with the NWD box loss (monkey-patched init_criterion).

Saves to runs/nwd_s<seed>/weights/best.pt. Compare against stock with eval_testset.py or
exp_multiarch_eval.py using --tags stock nwd.

  NWD_MODE=add NWD_LAMBDA=0.5 NWD_C=4.0 python train_nwd.py \
      --data data/arms/nearfar/data.yaml --imgsz 1280 --epochs 200 --batch 8 --device 0 --seed 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuttal_common import REPO_ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8s.pt", help="COCO-pretrained weights, as in the stock arm")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8,
                    help="so worker dataloader; dat 2 neu train treo (xem bay #5 CLAUDE.md)")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--runs", default=os.path.join(REPO_ROOT, "runs"))
    ap.add_argument("--nwd-lambda", type=float, default=None)
    ap.add_argument("--nwd-c", type=float, default=None)
    ap.add_argument("--nwd-mode", default=None, choices=[None, "add", "replace"])
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    # CLI overrides the environment variables when given
    if a.nwd_lambda is not None:
        os.environ["NWD_LAMBDA"] = str(a.nwd_lambda)
    if a.nwd_c is not None:
        os.environ["NWD_C"] = str(a.nwd_c)
    if a.nwd_mode is not None:
        os.environ["NWD_MODE"] = a.nwd_mode

    import ultralytics.nn.tasks as tasks
    from ultralytics import YOLO
    from nwd_loss import NWDDetectionLoss

    # replace DetectionModel.init_criterion
    tasks.DetectionModel.init_criterion = lambda self: NWDDetectionLoss(self)

    name = a.name or f"nwd_s{a.seed}"
    if a.resume:
        last = os.path.join(a.runs, name, "weights", "last.pt")
        if not os.path.exists(last):
            raise FileNotFoundError(f"Khong thay {last} de resume.")
        model = YOLO(last)
        model.train(resume=True)
    else:
        model = YOLO(a.model)
        model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
                    device=a.device, seed=a.seed, deterministic=True, patience=a.patience,
                    optimizer="SGD", lr0=0.01, project=a.runs, name=name, exist_ok=True,
                    prune=False)   # fork tich hop torch-pruning; tat de train chuan (giong arm stock)
    # no model.val(split=test): the real test set is the hand-labelled clips, evaluated separately
    best = os.path.join(a.runs, name, "weights", "best.pt")
    if os.path.exists(best):
        print(f"[{name}] DONE -> {best}")
    else:
        print(f"[{name}] !! no best.pt at {best}; see the training log above.")
    print(f"-> so sanh: python exp_multiarch_eval.py --labels-dir <LAB> --images-dir <IMG> "
          f"--runs {a.runs} --tags stock nwd --seeds 0 1 2 --imgsz 1280 --iou 0.3 --device 0")


if __name__ == "__main__":
    main()
