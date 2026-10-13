"""
Torchvision side of the comparison: SSD300-VGG16 and Faster R-CNN ResNet-50-FPN (v2).

The augmentation here re-implements EXACTLY the reduced Ultralytics policy used for YOLO
(random scale about centre with grey fill + box_candidates filter, HSV value gain, horizontal
flip), in the same order, so all three detectors see statistically identical training images.
"""
import copy
import json
import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch

from . import config as C

NUM_CLASSES = len(C.CLASSES) + 1  # + background (torchvision convention)


# ==========================================================================
# Models
# ==========================================================================
def build_model(name):
    """COCO-pretrained detector with a new head for the 4 damage classes + background (5 outputs, torchvision convention).
    Pretrained weights are always loaded (also when restoring a checkpoint) so that the
    module structure (e.g. FrozenBatchNorm in Faster R-CNN) is identical."""
    from torchvision.models import detection as D
    cfg = C.MODEL_CFG[name]
    if name == "ssd300_vgg16":
        from torchvision.models.detection import _utils as det_utils
        from torchvision.models.detection.ssd import SSDClassificationHead
        m = D.ssd300_vgg16(weights=D.SSD300_VGG16_Weights.COCO_V1,
                           trainable_backbone_layers=cfg["trainable_backbone_layers"],
                           score_thresh=C.INFER["conf_for_map"], nms_thresh=C.INFER["nms_iou"],
                           detections_per_img=C.INFER["max_det"], topk_candidates=400)
        in_ch = det_utils.retrieve_out_channels(m.backbone, (cfg["imgsz"], cfg["imgsz"]))
        n_anchors = m.anchor_generator.num_anchors_per_location()
        m.head.classification_head = SSDClassificationHead(in_ch, n_anchors, NUM_CLASSES)
    elif name == "fasterrcnn_r50_fpn_v2":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        m = D.fasterrcnn_resnet50_fpn_v2(weights=D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                                         trainable_backbone_layers=cfg["trainable_backbone_layers"],
                                         min_size=cfg["imgsz"], max_size=cfg["imgsz"],
                                         box_score_thresh=C.INFER["conf_for_map"],
                                         box_nms_thresh=C.INFER["nms_iou"],
                                         box_detections_per_img=C.INFER["max_det"])
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    else:
        raise ValueError(name)
    return m


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def param_groups(model, weight_decay):
    """Same rule as Ultralytics: weight decay on conv/linear weights only,
    not on biases or normalisation/scale parameters."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class ModelEMA:
    """Exponential moving average of weights (same formula as Ultralytics ModelEMA)."""

    def __init__(self, model, decay=C.EMA_DECAY, tau=C.EMA_TAU, updates=0):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = updates
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay_fn(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


# ==========================================================================
# Data
# ==========================================================================
def augment(img, boxes, labels, aug=C.AUG):
    """img RGB uint8 HxWx3, boxes (N,4) xyxy float32. Order = Ultralytics:
    random scale (random_perspective) -> HSV -> horizontal flip."""
    h, w = img.shape[:2]
    # 1) random scale about centre, grey (114) border, Ultralytics box_candidates filter
    s = random.uniform(1 - aug["scale"], 1 + aug["scale"])
    if aug["scale"] > 0:
        M = np.array([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
        if len(boxes):
            nb = boxes * s
            nb[:, [0, 2]] += (1 - s) * w / 2
            nb[:, [1, 3]] += (1 - s) * h / 2
            w1, h1 = nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]
            nb[:, [0, 2]] = nb[:, [0, 2]].clip(0, w)
            nb[:, [1, 3]] = nb[:, [1, 3]].clip(0, h)
            w2, h2 = nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]
            ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
            keep = (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (ar < 100)
            boxes, labels = nb[keep], labels[keep]
    # 2) HSV value (brightness) gain, hue/saturation untouched
    if aug["brightness"] > 0:
        g = random.uniform(1 - aug["brightness"], 1 + aug["brightness"])
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lut = np.clip(np.arange(256) * g, 0, 255).astype(np.uint8)
        hsv[..., 2] = cv2.LUT(hsv[..., 2], lut)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # 3) horizontal flip
    if random.random() < aug["hflip_p"]:
        img = np.ascontiguousarray(img[:, ::-1])
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img, boxes, labels


class CocoDetDataset(torch.utils.data.Dataset):
    def __init__(self, ann_json, img_dir, train=False):
        d = json.load(open(ann_json))
        self.images = sorted(d["images"], key=lambda im: im["id"])
        self.img_dir = img_dir
        self.train = train
        self.anns = defaultdict(list)
        for a in d["annotations"]:
            self.anns[a["image_id"]].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = cv2.imread(os.path.join(self.img_dir, im["file_name"]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        a = self.anns[im["id"]]
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in (b["bbox"] for b in a)],
                         dtype=np.float32).reshape(-1, 4)
        labels = np.array([b["category_id"] for b in a], dtype=np.int64)
        if self.train:
            img, boxes, labels = augment(img, boxes, labels)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        target = {"boxes": torch.from_numpy(boxes.astype(np.float32)).reshape(-1, 4),
                  "labels": torch.from_numpy(labels.astype(np.int64)),
                  "image_id": im["id"]}
        return t, target


def collate(batch):
    return tuple(zip(*batch))


# ==========================================================================
# Inference (same output format as YOLO side)
# ==========================================================================
@torch.no_grad()
def predict(model, ann_json, img_dir, batch_size=16, device="cuda", num_workers=C.NUM_WORKERS):
    ds = CocoDetDataset(ann_json, img_dir, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, collate_fn=collate)
    was_training = model.training
    model.eval()
    dets = []
    for imgs, tgts in dl:
        outs = model([im.to(device, non_blocking=True) for im in imgs])
        for o, t in zip(outs, tgts):
            b = o["boxes"].float().cpu().numpy()
            s = o["scores"].float().cpu().numpy()
            l = o["labels"].cpu().numpy()
            for (x1, y1, x2, y2), sc, lb in zip(b, s, l):
                if lb < 1:
                    continue
                dets.append({"image_id": int(t["image_id"]), "category_id": int(lb),
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             "score": float(sc)})
    if was_training:
        model.train()
    return dets


def load_for_inference(name, ckpt_path, device="cuda"):
    m = build_model(name)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["ema"] if "ema" in sd else sd["model"])
    return m.to(device).eval()
