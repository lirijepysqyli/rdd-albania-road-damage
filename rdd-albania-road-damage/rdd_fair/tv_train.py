"""
Training loop for the torchvision detectors, written to mirror the Ultralytics trainer:
  SGD(momentum 0.9, Nesterov) | no decay on bias/norm | linear warmup (>=100 it) |
  per-epoch cosine LR to 1% | AMP | grad-clip 10 | EMA | model selection on val mAP50-95.

Checkpoints (on Drive, written atomically):
  results/<model>/seed<k>/checkpoints/last.pt   every epoch, full state -> auto-resume
  results/<model>/seed<k>/checkpoints/best.pt   EMA weights of the selected epoch
"""
import csv
import math
import os
import random
import time

import numpy as np
import torch

from . import config as C
from . import prepare_data as P
from .fair_eval import Evaluator
from .tv_models import CocoDetDataset, ModelEMA, build_model, collate, count_params, param_groups, predict
from .utils import atomic_torch_save, load_json, save_json, seed_everything, worker_init_fn, write_run_info


def lr_factor(epoch, epochs=C.EPOCHS, lrf=C.FINAL_LR_FRAC):
    """Ultralytics cos_lr: 1 -> lrf over the run (evaluated per epoch)."""
    return ((1 - math.cos(epoch * math.pi / epochs)) / 2) * (lrf - 1) + 1


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def train(model_name, seed, device="cuda", val_batch=16):
    cfg = C.MODEL_CFG[model_name]
    ck_dir = C.run_dir(model_name, seed, "checkpoints")
    log_dir = C.run_dir(model_name, seed, "logs")
    done_flag = os.path.join(log_dir, "DONE")
    if os.path.exists(done_flag):
        print(f"[{model_name} seed {seed}] already finished -> skipping")
        return
    last_path, best_path = os.path.join(ck_dir, "last.pt"), os.path.join(ck_dir, "best.pt")
    log_csv = os.path.join(log_dir, "train_log.csv")

    seed_everything(seed)
    model = build_model(model_name).to(device)
    ds = CocoDetDataset(P.ann_path("train"), P.img_dir("train"), train=True)
    g = torch.Generator()
    accum = max(1, round(C.EFFECTIVE_BATCH / cfg["batch"]))
    base_lr = cfg["base_lr"]
    opt = torch.optim.SGD(param_groups(model, C.WEIGHT_DECAY), lr=base_lr,
                          momentum=C.MOMENTUM, nesterov=C.NESTEROV)
    scaler = torch.amp.GradScaler("cuda", enabled=C.AMP)
    ema = ModelEMA(model)
    evaluator = Evaluator(P.ann_path("val"))
    start_epoch, best, best_epoch = 0, -1.0, -1

    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.ema.load_state_dict(ck["ema"])
        ema.updates = ck["ema_updates"]
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, best_epoch = ck["epoch"], ck["best"], ck["best_epoch"]
        _set_rng_state(ck["rng"])
        print(f"[{model_name} seed {seed}] resumed from epoch {start_epoch} (best {best:.2f} @ {best_epoch})")
    else:
        write_run_info(model_name, seed, extra={"n_params": count_params(model), "grad_accum": accum,
                                                "base_lr": base_lr, "train_images": len(ds)})

    n_batches = math.ceil(len(ds) / cfg["batch"])
    n_warm = max(round(C.WARMUP_EPOCHS * n_batches), C.MIN_WARMUP_ITERS)
    print(f"[{model_name} seed {seed}] {len(ds)} train imgs | batch {cfg['batch']} x accum {accum} | "
          f"{n_batches} it/epoch | warmup {n_warm} it | lr {base_lr}")

    for epoch in range(start_epoch, C.EPOCHS):
        t0 = time.time()
        model.train()
        g.manual_seed(seed * 1000 + epoch)          # data order reproducible even after resume
        dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch"], shuffle=True, generator=g,
                                         num_workers=C.NUM_WORKERS, collate_fn=collate,
                                         worker_init_fn=worker_init_fn, pin_memory=True)
        target_lr = base_lr * lr_factor(epoch)
        for pg in opt.param_groups:
            pg["lr"] = target_lr
        opt.zero_grad(set_to_none=True)
        run_loss, run_parts, n_seen, n_skipped = 0.0, {}, 0, 0
        for i, (imgs, tgts) in enumerate(dl):
            ni = i + n_batches * epoch
            if ni <= n_warm:
                for pg in opt.param_groups:
                    pg["lr"] = float(np.interp(ni, [0, n_warm], [0.0, target_lr]))
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            tgts = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in tgts]
            with torch.autocast("cuda", enabled=C.AMP):
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == n_batches:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
            run_loss += loss.item()
            for k, v in loss_dict.items():
                run_parts[k] = run_parts.get(k, 0.0) + v.item()
            n_seen += 1
            if i % 200 == 0:
                print(f"  ep {epoch + 1}/{C.EPOCHS} it {i}/{n_batches} loss {loss.item():.4f} "
                      f"lr {opt.param_groups[0]['lr']:.5f}")
        train_time = time.time() - t0

        # ---- validation with the EMA model, using the shared evaluator ----
        t1 = time.time()
        dets = predict(ema.ema, P.ann_path("val"), P.img_dir("val"), batch_size=val_batch, device=device)
        m = evaluator.coco_metrics(dets)
        score = m[C.SELECTION_METRIC]
        is_best = score > best
        if is_best:
            best, best_epoch = score, epoch + 1
            atomic_torch_save({"ema": ema.ema.state_dict(), "epoch": epoch + 1, "val_metrics": m,
                               "model_name": model_name, "seed": seed}, best_path)
        atomic_torch_save({"model": model.state_dict(), "ema": ema.ema.state_dict(),
                           "ema_updates": ema.updates, "optimizer": opt.state_dict(),
                           "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best,
                           "best_epoch": best_epoch, "rng": _rng_state()}, last_path)

        row = {"epoch": epoch + 1, "lr": target_lr, "train_loss": run_loss / max(n_seen, 1),
               **{f"loss_{k}": v / max(n_seen, 1) for k, v in run_parts.items()},
               "val_mAP50": m["mAP50"], "val_mAP50_95": m["mAP50_95"],
               **{f"val_AP50_{c}": m["per_class"].get(c, {}).get("AP50", float("nan")) for c in C.CLASSES},
               "is_best": int(is_best), "skipped_nonfinite": n_skipped,
               "train_min": train_time / 60, "val_min": (time.time() - t1) / 60}
        new = not os.path.exists(log_csv)
        with open(log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[{model_name} seed {seed}] epoch {epoch + 1}: loss {row['train_loss']:.4f} | "
              f"val mAP50 {m['mAP50']:.2f} mAP50-95 {m['mAP50_95']:.2f} {'*BEST*' if is_best else ''} | "
              f"{train_time / 60:.1f}+{row['val_min']:.1f} min")

    save_json({"best_epoch": best_epoch, f"best_val_{C.SELECTION_METRIC}": best},
              os.path.join(log_dir, "selection.json"))
    open(done_flag, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[{model_name} seed {seed}] finished. Best epoch {best_epoch} (val {C.SELECTION_METRIC} {best:.2f})")
