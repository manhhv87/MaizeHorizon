# -*- coding: utf-8 -*-
"""Normalized Wasserstein Distance box loss for YOLOv8.

Rather than arguing that a small-object-friendly loss cannot beat the floor, this runs one. NWD is
designed for tiny objects, so if the far tier were loss-limited it should move.

Two modes:
  NWD_MODE=add      loss_box = (1-CIoU) + lambda*(1-NWD)   (default)
  NWD_MODE=replace  loss_box = (1-NWD)

Boxes are modelled as Gaussians N(center, diag((w/2)^2, (h/2)^2)):
  W2 = (dcx)^2 + (dcy)^2 + ((w1-w2)/2)^2 + ((h1-h2)/2)^2
  NWD = exp(-sqrt(W2)/C)
"""
import os

import torch

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


def _cfg():
    return {
        "lambda": float(os.environ.get("NWD_LAMBDA", "0.5")),
        "C": float(os.environ.get("NWD_C", "4.0")),
        "mode": os.environ.get("NWD_MODE", "add"),   # add | replace
    }


def normalized_wasserstein(pb, tb, C):
    """pb, tb: (N,4) xyxy in the same coordinate system; returns NWD (N,1) in [0,1]."""
    pcx = (pb[:, 0] + pb[:, 2]) * 0.5; pcy = (pb[:, 1] + pb[:, 3]) * 0.5
    pw = (pb[:, 2] - pb[:, 0]).clamp_min(1e-6); ph = (pb[:, 3] - pb[:, 1]).clamp_min(1e-6)
    tcx = (tb[:, 0] + tb[:, 2]) * 0.5; tcy = (tb[:, 1] + tb[:, 3]) * 0.5
    tw = (tb[:, 2] - tb[:, 0]).clamp_min(1e-6); th = (tb[:, 3] - tb[:, 1]).clamp_min(1e-6)
    w2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2 + ((pw - tw) * 0.5) ** 2 + ((ph - th) * 0.5) ** 2
    nwd = torch.exp(-torch.sqrt(w2 + 1e-7) / max(C, 1e-6))
    return nwd.unsqueeze(-1)


class NWDBboxLoss(BboxLoss):
    def __init__(self, reg_max=16, nwd_lambda=0.5, nwd_C=4.0, nwd_mode="add"):
        super().__init__(reg_max)
        self.nwd_lambda = nwd_lambda
        self.nwd_C = nwd_C
        self.nwd_mode = nwd_mode

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pb = pred_bboxes[fg_mask]
        tb = target_bboxes[fg_mask]
        iou = bbox_iou(pb, tb, xywh=False, CIoU=True)                 # (N,1)
        nwd = normalized_wasserstein(pb, tb, self.nwd_C)             # (N,1)
        if self.nwd_mode == "replace":
            loss_box = ((1.0 - nwd) * weight).sum() / target_scores_sum
        else:  # add / blend
            loss_box = (((1.0 - iou) + self.nwd_lambda * (1.0 - nwd)) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                     target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_box, loss_dfl


class NWDDetectionLoss(v8DetectionLoss):
    """Giong v8DetectionLoss nhung THAY bbox_loss = NWDBboxLoss (chi doi tin hieu HOI QUY)."""
    def __init__(self, model):
        super().__init__(model)
        c = _cfg()
        self.bbox_loss = NWDBboxLoss(self.reg_max, nwd_lambda=c["lambda"],
                                     nwd_C=c["C"], nwd_mode=c["mode"]).to(self.device)
        print(f"[NWD] bat NWDBboxLoss mode={c['mode']} lambda={c['lambda']} C={c['C']}")


# A full RFLA/NWD-RKA variant would also replace the assigner: subclass TaskAlignedAssigner and
# swap CIoU for NWD in get_box_metrics. That is more invasive and version-sensitive; the loss-term
# variant above already answers whether a tiny-object-aware signal lifts the far tier.
