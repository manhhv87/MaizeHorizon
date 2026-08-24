#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the stock and +Mint arms.

These are the two arms the paper reports as `stock` (human labels only) and `nearfar` (+Mint).
The defaults below are the values recorded in runs/<tag>_s<seed>/args.yaml.

  python train.py --arm stock   --seeds 0 1 2 --device 0
  python train.py --arm nearfar --seeds 0 1 2 --device 0

Other arms have their own entry points: train_tbxrd_stage2.py for the distillation arms,
train_nwd.py for NWD, exp_multiarch_train.py for the other architectures.
"""
import argparse
import os

ARMS = {
    "stock":   "data/arms/stock/data.yaml",      # human labels only
    "nearfar": "data/arms/nearfar/data.yaml",    # human labels plus minted positives
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default="stock")
    ap.add_argument("--data", default=None, help="override the arm's data.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--model", default="yolov8s.pt", help="COCO-pretrained weights")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8,
                    help="so worker cua dataloader. Mac dinh 8 la mac dinh cua ultralytics, "
                         "tuc gia tri da dung cho seed 0/1/2. Tren may nay 8 worker treo "
                         "vo han sau vai epoch (futex_wait_queue, GPU 0%%): deadlock fork "
                         "giua OpenCV va DataLoader. Dat 2 neu gap lai.")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None, help="ghi de ten thu muc run; mac dinh <arm>_s<seed>")
    a = ap.parse_args()

    from ultralytics import YOLO
    data = a.data or ARMS[a.arm]
    if not os.path.exists(data):
        raise SystemExit(f"data.yaml not found: {data}")

    for seed in a.seeds:
        name = f"{a.name or a.arm}_s{seed}"
        print(f"\n========== train {name} ==========", flush=True)
        YOLO(a.model).train(
            data=data, epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch,
            workers=a.workers,
            optimizer=a.optimizer, lr0=a.lr0, device=a.device, seed=seed,
            project=a.runs, name=name, exist_ok=True,
        )
        best = os.path.join(a.runs, name, "weights", "best.pt")
        if not os.path.exists(best):
            print(f"[{name}] !! no best.pt at {best}; see the training log above.")

    print("\n-> evaluate with eval_ap.py / eval_testset.py on the hand-labelled test clips")


if __name__ == "__main__":
    main()
