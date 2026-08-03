#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2: train the detector with cross-resolution feature distillation.

  python train_tbxrd_stage2.py --data data/arms/nearfar/data.yaml \
      --weights yolov8s.pt --imgsz 1280 --name distill_s0 \
      --mint-root data/mint --frames-root data/images --beta 0.5

Controls:
  --shuffle     pair each far region with a different plant's near view; any gain must vanish
  --synthetic   replace the near view with a blurred downsample; the real view should win

Geometric augmentation is disabled so the far box stays aligned with its RoI. Run
tbxrd_check_align.py first to confirm the alignment by eye.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=50,
                    help="early stopping patience; keep it high for distillation")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="distill_s0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mint-root", default="data/mint")
    ap.add_argument("--frames-root", default="data/images")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--level", type=int, default=0, help="0 = P3 (stride 8), the level that carries small plants")
    ap.add_argument("--near", type=int, default=128)
    ap.add_argument("--mode", default="cos", choices=["cos", "mse"])
    ap.add_argument("--shuffle", action="store_true", help="CONTROL: gan sai cap")
    ap.add_argument("--synthetic", action="store_true", help="CONTROL: near gia downsample")
    a = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils import tbxrd
    from ultralytics.models.yolo.detect.tbxrd_train import TBXRDDetectionTrainer

    tbxrd.CFG.update(mint_root=a.mint_root, frames_root=a.frames_root, beta=a.beta, level=a.level,
                     near=a.near, mode=a.mode, shuffle=a.shuffle, synthetic=a.synthetic)

    YOLO(a.weights).train(
        trainer=TBXRDDetectionTrainer, data=a.data, epochs=a.epochs, patience=a.patience, imgsz=a.imgsz,
        batch=a.batch, device=a.device, name=a.name, seed=a.seed, project="runs", prune=False,
        # geometric augmentation must be off so the far box stays aligned with its RoI
        mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
        fliplr=0.0, flipud=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    )


if __name__ == "__main__":
    main()
