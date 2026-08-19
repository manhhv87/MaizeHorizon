# -*- coding: utf-8 -*-
"""
sgc_ext.py -- phần loss-side cho YOLO-SGC-PSF: PSF supervision + PILA assignment.
Tự chứa. Bật qua biến môi trường (xem get_cfg). Dùng bởi train_sgc.py.
"""
import math
import os

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
from ultralytics.nn.modules.sgc import _SGC_STATE


# ----------------------------- config (env) -----------------------------
def _b(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def get_cfg():
    return {
        "psf_mode": os.environ.get("SGC_PSF_MODE", "size"),     # none|row|size (size: aug-invariant)
        "pila_mode": os.environ.get("SGC_PILA_MODE", "soft"),   # off|soft|hard
        "psf_shuffle": _b(os.environ.get("SGC_PSF_SHUFFLE", "0")),
        "psf_weight": float(os.environ.get("SGC_PSF_W", "0.1")),
        "psf_gain": float(os.environ.get("SGC_PSF_GAIN", "1.0")),  # extra multiplier so S converges
        "pila_sigma": float(os.environ.get("SGC_PILA_SIGMA", "0.7")),
        "pila_warmup": int(os.environ.get("SGC_PILA_WARMUP", "3")),  # epoch PILA starts
        "pila_ramp": int(os.environ.get("SGC_PILA_RAMP", "5")),      # epochs to ramp gate 0->1 (soft)
        "pila_lmin": float(os.environ.get("SGC_PILA_LMIN", "3")),
        "pila_lmax": float(os.environ.get("SGC_PILA_LMAX", "5")),
    }


# ----------------------------- helpers -----------------------------
def sample_field_at_points(field, pts_norm):
    grid = (pts_norm * 2.0 - 1.0).unsqueeze(1).to(field.dtype)  # (B,1,N,2); khớp dtype (AMP half)
    out = F.grid_sample(field, grid, mode="bilinear", align_corners=False, padding_mode="border")
    return out[:, 0, 0, :]  # (B,N)


def gt_centers_norm(gt_bboxes, img_w, img_h):
    cx = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5 / img_w
    cy = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5 / img_h
    return torch.stack([cx, cy], dim=-1).clamp(0.0, 1.0)


def normalize_logsize(gt_bboxes, lo=None, hi=None):
    w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_min(1.0)
    h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_min(1.0)
    s = torch.log(torch.sqrt(w * h))
    lo = math.log(8.0) if lo is None else lo
    hi = math.log(640.0) if hi is None else hi
    return ((s - lo) / (hi - lo)).clamp(0.0, 1.0)


def build_anc_levels(strides, num_per_level, device):
    levels = []
    for s, n in zip(list(strides), list(num_per_level)):
        levels.append(torch.log2(torch.tensor(float(s))).repeat(int(n)))
    return torch.cat(levels).to(device)


def level_gate(gt_scale, anc_levels, l_min, l_max, sigma, hard):
    l_pref = l_min + (l_max - l_min) * gt_scale.clamp(0, 1)          # (B,N)
    diff = anc_levels.view(1, 1, -1) - l_pref.unsqueeze(-1)          # (B,N,A)
    if hard:
        return (diff.round().abs() < 0.5).float().clamp_min(1e-3)
    return torch.exp(-(diff ** 2) / (2.0 * sigma * sigma))


def psf_loss(S, mode, weight, shuffle=False, centers=None, logsize=None, mask=None):
    if mode == "none":
        return S.sum() * 0.0
    if mode == "row":
        b, _, h, w = S.shape
        tgt = torch.linspace(0, 1, h, device=S.device, dtype=S.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        if shuffle:
            tgt = tgt[:, :, torch.randperm(h, device=S.device), :]
        return weight * F.smooth_l1_loss(S, tgt)
    # size
    pred = sample_field_at_points(S, centers)
    m = (mask > 0) if mask is not None else torch.ones_like(logsize, dtype=torch.bool)
    if m.sum() == 0:
        return S.sum() * 0.0
    tgt = logsize
    if shuffle:
        flat = tgt[m]
        tgt = tgt.clone()
        tgt[m] = flat[torch.randperm(flat.numel(), device=S.device)]
    return weight * F.smooth_l1_loss(pred[m], tgt[m])


# ----------------------------- assigner -----------------------------
class PILAAssigner(TaskAlignedAssigner):
    def __init__(self, *a, l_min=3.0, l_max=5.0, sigma=0.7, hard=False, **k):
        super().__init__(*a, **k)
        self.l_min, self.l_max, self.sigma, self.hard = l_min, l_max, sigma, hard
        self._gate = None
        self._ramp = 1.0   # soft warmup factor in [0,1], set by the loss each step

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt,
                anc_levels=None, gt_scale=None):
        self._gate = None
        if anc_levels is not None and gt_scale is not None:
            self._gate = level_gate(gt_scale, anc_levels.to(pd_scores.device),
                                    self.l_min, self.l_max, self.sigma, self.hard)
        return super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                                                      mask_in_gts * mask_gt)
        # PILA gates ONLY the top-k SELECTION path (a copy). The returned align_metric/overlaps
        # stay UNGATED, so TAL's target_scores normalization is untouched -> no non-stationary
        # cls targets -> no oscillation. (Soft ramp blends the gate toward identity early on.)
        am_topk = align_metric
        if self._gate is not None:
            r = float(self._ramp)
            gate = (1.0 - r) + r * self._gate.to(align_metric.dtype)   # r=0 -> identity, r=1 -> full
            am_topk = align_metric * gate
        mask_topk = self.select_topk_candidates(am_topk, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps


def _find_psfhead(model):
    for m in model.modules():
        if type(m).__name__ == "PSFHead":
            return m
    return None


# ----------------------------- loss -----------------------------
class SGCDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        c = get_cfg()
        self.cfg = c
        self.epoch = 0
        self._psfhead = _find_psfhead(model)
        self.assigner = PILAAssigner(
            topk=getattr(self.assigner, "topk", 10), num_classes=self.nc,
            alpha=getattr(self.assigner, "alpha", 0.5), beta=getattr(self.assigner, "beta", 6.0),
            l_min=c["pila_lmin"], l_max=c["pila_lmax"], sigma=c["pila_sigma"],
            hard=(c["pila_mode"] == "hard"))

    def __call__(self, preds, batch):
        c = self.cfg
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, psf
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        bs = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        S = _SGC_STATE.get("S", None)
        if S is not None:
            S = S.float()  # PSF/PILA tính ở fp32 -> tránh lỗi "Half vs Float" lúc AMP/val
        extra = {}
        if c["pila_mode"] != "off" and self.epoch >= c["pila_warmup"] and S is not None:
            nl = [int(f.shape[-1] * f.shape[-2]) for f in feats]
            centers = gt_centers_norm(gt_bboxes, float(imgsz[1]), float(imgsz[0]))
            # Soft warmup: gate blends from identity (epoch == pila_warmup) to full over
            # pila_ramp epochs -> no abrupt assignment switch -> stabler. (get_pos_mask uses _ramp.)
            self.assigner._ramp = min(max((self.epoch - c["pila_warmup"]) / max(c["pila_ramp"], 1), 0.0), 1.0)
            # S đã được PSF giám sát về [0,1] (row/size); level_gate tự clamp(0,1).
            # KHÔNG bọc sigmoid ở đây: sigmoid([0,1])=~[0.5,0.73] -> l_pref kẹt ~[4.0,4.46]
            # khiến PILA không thể định tuyến far->P3 / near->P5 (defeat cơ chế).
            extra = dict(anc_levels=build_anc_levels(self.stride, nl, feats[0].device),
                         gt_scale=sample_field_at_points(S, centers))

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt, **extra)

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes = target_bboxes / stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes,
                                              target_scores, target_scores_sum, fg_mask)

        if c["psf_mode"] != "none" and S is not None:
            if c["psf_mode"] == "size":
                centers = gt_centers_norm(gt_bboxes, float(imgsz[1]), float(imgsz[0]))
                loss[3] = psf_loss(S, "size", c["psf_weight"], c["psf_shuffle"],
                                   centers=centers, logsize=normalize_logsize(gt_bboxes),
                                   mask=mask_gt.squeeze(-1))
            else:
                loss[3] = psf_loss(S, "row", c["psf_weight"], c["psf_shuffle"])
            loss[3] *= c["psf_gain"]   # extra multiplier so S converges before PILA leans on it

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * bs, loss.detach()
