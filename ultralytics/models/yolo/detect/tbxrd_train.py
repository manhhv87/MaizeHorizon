# -*- coding: utf-8 -*-
"""
TBXRDDetectionTrainer -- TB-XRD Stage 2 (cross-resolution feature distillation).

Chi can: model.train(trainer=TBXRDDetectionTrainer, ...). Cau hinh qua ultralytics.utils.tbxrd.CFG
(entrypoint cap nhat truoc khi train) de tranh truyen kwarg la cho ultralytics.

Khong sua head/loss/model: dung forward_pre_hook lay feature P3 cua student & teacher, projector
chi vao optimizer param-group (khong vao state_dict -> inference giu stock), criterion 4-thanh-phan.

!! BAT BUOC tat aug hinh hoc (mosaic/mixup/fliplr/translate/scale/perspective/shear/degrees = 0)
   de letterbox la phep bien hinh duy nhat -> map_far_xyxy_to_input chinh xac. Entrypoint da set.
"""
import os

import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.tbxrd import (
    CFG, TBXRDDetectionLoss, FeatureProjector,
    load_pairs_index, build_near_crop_cache, map_far_xyxy_to_input,
)


class TBXRDDetectionTrainer(DetectionTrainer):
    """DetectionTrainer + EMA-teacher cross-resolution feature distillation."""

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self.level = int(CFG["level"]); self.beta = float(CFG["beta"])
        self.near_size = int(CFG["near"]); self.feat_mode = CFG["mode"]
        self.shuffle_pairs = bool(CFG["shuffle"]); self.synthetic_near = bool(CFG["synthetic"])
        self._student_feats, self._teacher_feats = {}, {}
        self.projector = None
        self.pairs_index, self.near_cache = {}, {}

        if self.ema is None:
            LOGGER.warning("[TB-XRD] khong co EMA -> bo distillation (chi train detection).")
            return

        m = de_parallel(self.model)
        detect = m.model[-1]
        self.roi_stride = float(detect.stride[self.level])   # KHONG ghi de self.stride cua trainer (vo multi_scale)
        # pre-hook lay P3/P4/P5 TRUOC khi Detect.forward cat in-place. Luu module de go/gan lai quanh save_model
        # (hook la lambda -> torch.save pickle model+EMA se vap 'Can't pickle local lambda').
        self._student_detect = detect
        self._teacher_detect = self.ema.ema.model[-1]
        self._student_hook = self._teacher_hook = None
        self._install_hooks()

        # so kenh C qua 1 forward gia (EVAL mode -> khong ban BN running stats voi batch zeros)
        was_training = m.training
        m.eval()
        with torch.no_grad():
            m(torch.zeros(1, 3, self.args.imgsz, self.args.imgsz, device=self.device))
        if was_training:
            m.train()
        C = self._student_feats["p"][self.level].shape[1]
        self.projector = FeatureProjector(C).to(self.device).train()
        # initial_lr BAT BUOC (warmup loop doc x['initial_lr'] -> thieu se KeyError) + noi vao scheduler
        self.optimizer.add_param_group(
            {"params": self.projector.parameters(), "lr": self.args.lr0,
             "initial_lr": self.args.lr0, "weight_decay": 0.0})
        self.scheduler.base_lrs.append(self.args.lr0)
        self.scheduler.lr_lambdas.append(self.scheduler.lr_lambdas[0])

        # nap pairs + cache near crop (1 lan)
        self.pairs_index = load_pairs_index(CFG["mint_root"])
        if not self.synthetic_near:
            self.near_cache, n_ok, n_miss = build_near_crop_cache(
                self.pairs_index, CFG["frames_root"], self.near_size)
        else:
            n_ok = n_miss = 0
        LOGGER.info(f"[TB-XRD] frames voi pair={len(self.pairs_index)} | near ok={n_ok} miss={n_miss} "
                    f"| C={C} roi_stride={self.roi_stride} beta={self.beta} level={self.level} "
                    f"shuffle={self.shuffle_pairs} synthetic={self.synthetic_near}")
        # LOI CUNG neu khong co tin hieu distill -> tranh am tham tao arm 'distill' = baseline (sai ket qua)
        if not self.pairs_index:
            raise RuntimeError("[TB-XRD] khong nap duoc pair tu CFG['mint_root']! Distill vo nghia. Kiem mint_root + cwd.")
        if (not self.synthetic_near) and n_ok == 0:
            raise RuntimeError("[TB-XRD] near-crop cache RONG (CFG['frames_root'] sai?) -> L_feat luon 0. Dung.")

        # cai criterion 4-vector cho CA student VA teacher (validate dung EMA-teacher -> phai cung 4-vector)
        crit = TBXRDDetectionLoss(m); crit.tbxrd = self; m.criterion = crit
        self.ema.ema.criterion = TBXRDDetectionLoss(self.ema.ema)   # khong .tbxrd -> tra (box,cls,dfl,0)

    def _install_hooks(self):
        """Dang ky pre-hook lay feature P3 cua student + teacher (lambda -> khong pickle duoc)."""
        if self._student_detect is None:
            return
        self._student_hook = self._student_detect.register_forward_pre_hook(
            lambda mod, inp: self._student_feats.__setitem__("p", list(inp[0])))
        self._teacher_hook = self._teacher_detect.register_forward_pre_hook(
            lambda mod, inp: self._teacher_feats.__setitem__("p", list(inp[0])))

    def _remove_hooks(self):
        """Go pre-hook (truoc torch.save) de model/EMA pickle duoc."""
        for name in ("_student_hook", "_teacher_hook"):
            h = getattr(self, name, None)
            if h is not None:
                h.remove(); setattr(self, name, None)

    def save_model(self):
        """Go lambda-hook quanh torch.save (neu khong se 'Can't pickle local lambda'), gan lai sau."""
        had = getattr(self, "_student_hook", None) is not None
        self._remove_hooks()
        try:
            super().save_model()
        finally:
            if had:
                self._install_hooks()

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.projector is None or not self.pairs_index:
            batch["distill"] = None
            return batch
        boxes, bidx, crops = [], [], []
        imgsz = self.args.imgsz
        for j, fpath in enumerate(batch["im_file"]):
            plist = self.pairs_index.get(os.path.basename(fpath))
            if not plist:
                continue
            cc = self.near_cache.get(os.path.basename(fpath), [])
            H, W = batch["ori_shape"][j]
            for pi, p in enumerate(plist):
                if not self.synthetic_near:
                    nc = cc[pi] if pi < len(cc) else None
                    if nc is None:
                        continue
                    crops.append(nc)
                boxes.append(map_far_xyxy_to_input(p["far"]["xyxy"], (H, W), imgsz))
                bidx.append(j)
        if not boxes:
            batch["distill"] = None
            return batch
        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        bidx_t = torch.tensor(bidx, dtype=torch.long, device=self.device)
        if self.synthetic_near:
            near_b = self._synthetic_near(batch["img"], boxes_t, bidx_t)   # CONTROL: near gia tu far-crop mo
        else:
            near_b = torch.stack(crops).to(self.device).float() / 255.0
        if self.shuffle_pairs:                                              # CONTROL: gán SAI cặp
            if near_b.shape[0] < 2:
                batch["distill"] = None        # 1 cặp -> không thể gán sai -> bỏ (không để tín hiệu thật lọt vào control)
                return batch
            near_b = torch.roll(near_b, 1, 0)  # mỗi far lấy near của cây KHÁC -> đảm bảo gán sai (derangement)
        batch["distill"] = {"boxes": boxes_t, "batch_idx": bidx_t, "near": near_b}
        return batch

    def _synthetic_near(self, img, boxes, bidx):
        """near = far-crop downsample x4 roi upsample (view xa mo) -> teacher input gia."""
        import torch.nn.functional as F
        Himg, Wimg = img.shape[2], img.shape[3]
        out = []
        for k in range(boxes.shape[0]):
            j = int(bidx[k]); x1, y1, x2, y2 = boxes[k].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(Wimg, int(x2)), min(Himg, int(y2))
            h, w = y2 - y1, x2 - x1
            if h < 2 or w < 2:
                out.append(torch.zeros(3, self.near_size, self.near_size, device=img.device)); continue
            crop = img[j:j + 1, :, y1:y2, x1:x2]
            # downsample ~4x nhung CLAMP >=1: box far RAT HEP (w 2-3px) * 0.25 -> 0 -> IndexError o interpolate sau
            small = F.interpolate(crop, size=(max(1, h // 4), max(1, w // 4)), mode="bilinear", align_corners=False)
            up = F.interpolate(small, size=(self.near_size, self.near_size), mode="bilinear", align_corners=False)
            out.append(up[0])
        return torch.stack(out)

    def get_validator(self):
        v = super().get_validator()
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "feat_loss")
        return v
