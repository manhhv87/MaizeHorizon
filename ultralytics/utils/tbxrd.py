# -*- coding: utf-8 -*-
"""
tbxrd.py -- TB-XRD Stage 2: cross-resolution feature distillation pieces.

Faithful core (BYOL-style): during far-frame detection training, for each minted plant
  (1) RoIAlign the STUDENT P3 feature at the far box -> pool -> train-only projector -> s;
  (2) forward the NEAR crop through the EMA TEACHER (self.ema.ema) -> global-pool -> t (stop-grad);
  (3) L_feat = mean(1 - cos(s, sg(t))).
Inference stays byte-for-byte stock: the projector is added only as an optimizer param-group
(never in model.state_dict / EMA / checkpoint); hooks + teacher are trainer-owned.

CFG is a module-level dict the entrypoint updates before train() (avoids passing unknown kwargs).
"""
import os
import glob
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss

# Cau hinh, entrypoint cap nhat (tbxrd.CFG.update(...)) truoc khi train.
CFG = {
    "mint_root": "mint",                                # thu muc chua <clip>/pairs.jsonl
    "frames_root": "datasets/corn_plant/to_label",      # de tim anh NEAR (frames_root/<clip>/<near.frame>)
    "beta": 0.5,                                         # trong so L_feat
    "level": 0,                                          # 0=P3 (stride 8), 1=P4, 2=P5
    "near": 128,                                         # kich thuoc crop near (teacher input)
    "mode": "cos",                                       # 'cos' hoac 'mse'
    "shuffle": False,                                    # CONTROL: gan sai cap (near hoan vi)
    "synthetic": False,                                  # CONTROL: near = far-crop downsample (gia)
}


def load_pairs_index(mint_root):
    """Quet mint_root/*/pairs.jsonl -> {far_frame_basename: [pair,...]} (giu thu tu)."""
    index = {}
    for pj in glob.glob(os.path.join(mint_root, "*", "pairs.jsonl")):
        with open(pj, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except Exception:
                    continue
                fb = os.path.basename(p["far"]["frame"])
                index.setdefault(fb, []).append(p)
    return index


def build_near_crop_cache(pairs_index, frames_root, near_size=128):
    """Crop near.xyxy tu near.frame (frames_root/<clip>/) -> uint8 RGB CHW tensor.
    Tra (cache{far_basename: [tensor|None,...]} cung thu tu pairs_index, n_ok, n_miss)."""
    import cv2
    cache = {}
    n_ok = n_miss = 0
    for fb, plist in pairs_index.items():
        crops = []
        for p in plist:
            clip = p.get("clip", "")
            near_fb = os.path.basename(p["near"]["frame"])
            ip = os.path.join(frames_root, clip, near_fb)
            im = cv2.imread(ip)
            if im is None:
                crops.append(None); n_miss += 1; continue
            H, W = im.shape[:2]
            x1, y1, x2, y2 = p["near"]["xyxy"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W, int(x2)), min(H, int(y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                crops.append(None); n_miss += 1; continue
            crop = cv2.resize(im[y1:y2, x1:x2], (near_size, near_size))[:, :, ::-1]  # BGR->RGB
            crops.append(torch.from_numpy(crop.copy()).permute(2, 0, 1).contiguous())  # uint8 CHW
            n_ok += 1
        cache[fb] = crops
    return cache, n_ok, n_miss


class FeatureProjector(nn.Module):
    """BYOL predictor tren phia STUDENT (train-only, khong vao state_dict cua model)."""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, c))

    def forward(self, x):
        return self.net(x)


def map_far_xyxy_to_input(xyxy, ori_shape, imgsz):
    """Box tu px ANH GOC -> px ANH INPUT (letterbox xac dinh; CHI dung khi tat aug hinh hoc).
    ori_shape=(H,W); imgsz=int (vuong)."""
    H, W = ori_shape
    gain = min(imgsz / H, imgsz / W)
    pad_x = (imgsz - W * gain) / 2.0
    pad_y = (imgsz - H * gain) / 2.0
    x1, y1, x2, y2 = xyxy
    return [x1 * gain + pad_x, y1 * gain + pad_y, x2 * gain + pad_x, y2 * gain + pad_y]


def roi_pool_student(feat, boxes_xyxy, batch_idx, stride, out=7):
    """Mean-pool feature student trong tung box far (toa do INPUT px) -> (K,C).

    KHONG dung torchvision.ops.roi_align: tren env co torch<->torchvision LECH, kernel native khong nap
    duoc -> roi_align roi xuong ban Python `_roi_align` materialize tensor 6D [K,C,PH,PW,IY,IX] -> CUDA OOM.
    Voi distill cosine ta chi can VECTOR DAC TRUNG TRUNG BINH trong moi box (roi_align+adaptive_avg_pool2d(.,1)
    cung chi ra trung binh) -> mean-pool truc tiep tren feature: dung gradient, ton ~0 bo nho, khong phu thuoc env.
    """
    C = feat.shape[1]
    K = int(boxes_xyxy.shape[0])
    if K == 0:
        return feat.new_zeros((0, C))
    feat = feat.float()                                              # fp32: on dinh duoi AMP fp16
    _, _, H, W = feat.shape
    sc = 1.0 / float(stride)
    b_cpu = batch_idx.detach().to("cpu").tolist()                    # 1 lan chuyen CPU -> tranh sync moi vong lap
    xy = (boxes_xyxy.detach().float() * sc).to("cpu").tolist()       # box -> toa do feature-map (P3)
    vecs = []
    for k in range(K):
        b = int(b_cpu[k]); x1, y1, x2, y2 = xy[k]
        x1i = max(0, min(W - 1, int(x1))); y1i = max(0, min(H - 1, int(y1)))
        x2i = max(x1i + 1, min(W, int(x2) + 1)); y2i = max(y1i + 1, min(H, int(y2) + 1))   # >=1 cell, khong rong
        vecs.append(feat[b, :, y1i:y2i, x1i:x2i].mean(dim=(1, 2)))   # (C,) - gradient chay qua feat
    return torch.stack(vecs, dim=0)                                  # (K,C)


@torch.no_grad()
def teacher_near_embed(ema_model, near_crops, teacher_feats, level):
    """Forward near-crop qua EMA teacher; teacher_feats['p'] do pre-hook nap. Tra (K,C) detached."""
    if near_crops.numel() == 0:
        return near_crops.new_zeros((0, 1))
    teacher_feats.clear()
    _ = ema_model(near_crops)                         # kich hoat pre-hook tren Detect cua teacher
    p = teacher_feats["p"][level]                     # (K,C,h,w)
    return F.adaptive_avg_pool2d(p, 1).flatten(1).detach()  # (K,C)


def feat_distill_loss(student_vec, teacher_vec, mode="cos"):
    """student_vec:(K,C) grad; teacher_vec:(K,C) detached. Scalar."""
    if student_vec.shape[0] == 0:
        return student_vec.sum() * 0.0
    s = student_vec.float()
    t = teacher_vec.float()
    if mode == "mse":
        return F.mse_loss(F.normalize(s, dim=1), F.normalize(t, dim=1))
    s = F.normalize(s, dim=1); t = F.normalize(t, dim=1)
    return (1.0 - (s * t).sum(1)).mean()


class TBXRDDetectionLoss(v8DetectionLoss):
    """v8 detection loss + TB-XRD cross-resolution feature distillation (Stage 2).

    Trainer gan `.tbxrd` (chinh la trainer) mang: _student_feats / _teacher_feats (pre-hook nap),
    projector, ema, beta, level, stride, feat_mode. Tra loss 4 thanh phan -> trainer tu mo rong tloss/log.
    """

    def __call__(self, preds, batch):
        loss_bw, loss_log = super().__call__(preds, batch)            # moi cai (3,): box,cls,dfl  (loss_bw da *batch_size)
        tx = getattr(self, "tbxrd", None)
        d = batch.get("distill") if isinstance(batch, dict) else None
        if tx is None or getattr(tx, "projector", None) is None or d is None or d["boxes"].shape[0] == 0:
            z = loss_bw.new_zeros(1)
            return torch.cat([loss_bw, z]), torch.cat([loss_log, z])
        feat = tx._student_feats["p"][tx.level]                       # (B,C,h,w) co grad cua student
        s = tx.projector(roi_pool_student(feat, d["boxes"], d["batch_idx"], tx.roi_stride))
        t = teacher_near_embed(tx.ema.ema, d["near"], tx._teacher_feats, tx.level)
        l_feat = feat_distill_loss(s, t, mode=tx.feat_mode)
        bs = batch["img"].shape[0]
        feat_bw = (tx.beta * l_feat * bs).view(1)                     # cung *batch_size nhu det_bw
        feat_log = (tx.beta * l_feat).detach().view(1)
        return torch.cat([loss_bw, feat_bw]), torch.cat([loss_log, feat_log])
