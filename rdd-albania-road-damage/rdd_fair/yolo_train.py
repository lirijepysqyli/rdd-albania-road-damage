"""
YOLOv8m side of the comparison.

Ultralytics is configured to follow the same protocol as the torchvision models:
mosaic / mixup / translate / rotation / HSV hue+sat / vertical flip OFF; only random scale 0.1,
HSV value 0.2 and horizontal flip 0.5 remain. nbs=16 so the effective batch is 16 with the same
weight decay. Albumentations must be UNINSTALLED in this runtime (Ultralytics would otherwise
silently add blur / CLAHE / grayscale).

Model selection does NOT use Ultralytics' internal 'fitness'. After every epoch a callback
evaluates the freshly saved EMA weights (last.pt) on val with fair_eval and copies the best one to
results/yolov8m/seed<k>/checkpoints/best.pt.
"""
import csv
import os
import shutil
import time

import numpy as np

from . import config as C
from . import prepare_data as P
from .fair_eval import Evaluator
from .utils import load_json, save_json, seed_everything, write_run_info

MODEL = "yolov8m"


def train_args(seed, data_yaml):
    cfg = C.MODEL_CFG[MODEL]
    return dict(
        data=data_yaml, epochs=C.EPOCHS, imgsz=cfg["imgsz"], batch=cfg["batch"], nbs=C.EFFECTIVE_BATCH,
        seed=seed, deterministic=True, workers=C.NUM_WORKERS, amp=C.AMP, pretrained=True,
        optimizer="SGD", lr0=cfg["base_lr"], lrf=C.FINAL_LR_FRAC, momentum=C.MOMENTUM,
        weight_decay=C.WEIGHT_DECAY, cos_lr=True,
        warmup_epochs=C.WARMUP_EPOCHS, warmup_momentum=C.MOMENTUM, warmup_bias_lr=0.0,
        # ---- shared augmentation policy ----
        hsv_h=0.0, hsv_s=0.0, hsv_v=C.AUG["brightness"], degrees=0.0, translate=0.0,
        scale=C.AUG["scale"], shear=0.0, perspective=0.0, flipud=0.0, fliplr=C.AUG["hflip_p"],
        mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
        rect=False, multi_scale=False, single_cls=False, cache=False,
        # ---- bookkeeping ----
        patience=C.EPOCHS + 1, val=False, plots=True, save=True, save_period=-1,
        project=C.run_dir(MODEL, seed), name="ultralytics", exist_ok=True,
        iou=C.INFER["nms_iou"], max_det=C.INFER["max_det"],
    )


def predict(model, ann_json, img_dir, batch=16, conf=None):
    """Same output format as tv_models.predict. Class-wise NMS, FP32."""
    import json
    d = json.load(open(ann_json))
    name2id = {im["file_name"]: im["id"] for im in d["images"]}
    paths = [os.path.join(img_dir, im["file_name"]) for im in sorted(d["images"], key=lambda x: x["id"])]
    cfg = C.MODEL_CFG[MODEL]
    dets = []
    for k in range(0, len(paths), 256):
        for r in model.predict(paths[k:k + 256], imgsz=cfg["imgsz"],
                               conf=C.INFER["conf_for_map"] if conf is None else conf,
                               iou=C.INFER["nms_iou"], max_det=C.INFER["max_det"], agnostic_nms=False,
                               half=False, batch=batch, verbose=False, stream=True):
            iid = name2id[os.path.basename(r.path)]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            sc = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, s in zip(xyxy, cls, sc):
                dets.append({"image_id": iid, "category_id": int(c) + 1,
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             "score": float(s)})
    return dets


def _make_callback(seed, evaluator):
    from ultralytics import YOLO
    ck_dir = C.run_dir(MODEL, seed, "checkpoints")
    log_dir = C.run_dir(MODEL, seed, "logs")
    state_path = os.path.join(log_dir, "selection_state.json")
    log_csv = os.path.join(log_dir, "train_log.csv")

    def on_fit_epoch_end(trainer):
        import torch
        t1 = time.time()
        epoch = trainer.epoch + 1
        state = load_json(state_path, {"best": -1.0, "best_epoch": -1, "evaluated_epochs": []})
        if epoch in state["evaluated_epochs"]:
            return
        m_eval = YOLO(str(trainer.last))                   # EMA weights saved this epoch
        dets = predict(m_eval, P.ann_path("val"), P.img_dir("val"))
        m = evaluator.coco_metrics(dets)
        score = m[C.SELECTION_METRIC]
        is_best = score > state["best"]
        if is_best:
            tmp = os.path.join(ck_dir, "best.pt.tmp")
            shutil.copy2(str(trainer.last), tmp)
            os.replace(tmp, os.path.join(ck_dir, "best.pt"))
            state.update(best=score, best_epoch=epoch)
        state["evaluated_epochs"].append(epoch)
        save_json(state, state_path)
        losses = trainer.label_loss_items(trainer.tloss, prefix="train") if trainer.tloss is not None else {}
        row = {"epoch": epoch, "lr": trainer.lr.get("lr/pg0", np.nan) if isinstance(trainer.lr, dict) else np.nan,
               **{k.replace("/", "_"): float(v) for k, v in losses.items()},
               "val_mAP50": m["mAP50"], "val_mAP50_95": m["mAP50_95"],
               **{f"val_AP50_{c}": m["per_class"].get(c, {}).get("AP50", float("nan")) for c in C.CLASSES},
               "is_best": int(is_best), "val_min": (time.time() - t1) / 60}
        new = not os.path.exists(log_csv)
        with open(log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[yolov8m seed {seed}] epoch {epoch}: val mAP50 {m['mAP50']:.2f} "
              f"mAP50-95 {m['mAP50_95']:.2f} {'*BEST*' if is_best else ''}")
        del m_eval
        torch.cuda.empty_cache()

    return on_fit_epoch_end


def train(seed, data_yaml):
    import torch
    from ultralytics import YOLO
    log_dir = C.run_dir(MODEL, seed, "logs")
    done_flag = os.path.join(log_dir, "DONE")
    if os.path.exists(done_flag):
        print(f"[yolov8m seed {seed}] already finished -> skipping")
        return
    seed_everything(seed)
    evaluator = Evaluator(P.ann_path("val"))
    cb = _make_callback(seed, evaluator)
    last = os.path.join(C.run_dir(MODEL, seed), "ultralytics", "weights", "last.pt")

    finished = False
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("epoch", 0) == -1:          # Ultralytics strips last.pt when training completes
            finished = True
        else:
            print(f"[yolov8m seed {seed}] resuming from epoch {ck['epoch'] + 1}")
            model = YOLO(last)
            model.add_callback("on_fit_epoch_end", cb)
            model.train(resume=True)
            finished = True
    if not finished:
        model = YOLO(C.MODEL_CFG[MODEL]["weights"])
        n_params = sum(p.numel() for p in model.model.parameters())
        write_run_info(MODEL, seed, extra={"n_params": n_params, "ultralytics_args": train_args(seed, data_yaml),
                                           "base_lr": C.MODEL_CFG[MODEL]["base_lr"], "grad_accum": 1})
        model.add_callback("on_fit_epoch_end", cb)
        model.train(**train_args(seed, data_yaml))

    state = load_json(os.path.join(log_dir, "selection_state.json"), {})
    save_json({"best_epoch": state.get("best_epoch"), f"best_val_{C.SELECTION_METRIC}": state.get("best")},
              os.path.join(log_dir, "selection.json"))
    open(done_flag, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[yolov8m seed {seed}] finished. Best epoch {state.get('best_epoch')} "
          f"(val {C.SELECTION_METRIC} {state.get('best', float('nan')):.2f})")
