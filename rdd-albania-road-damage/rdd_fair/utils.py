"""Reproducibility helpers shared by all notebooks."""
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------
def seed_everything(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # warn_only: some detection ops (e.g. roi_align backward) have no deterministic kernel
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def worker_init_fn(worker_id):
    import torch
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_torch_save(obj, path):
    """Save to a temp file then rename, so a Colab disconnect can't corrupt the checkpoint."""
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Environment capture / version pinning
# --------------------------------------------------------------------------
PINNED_PACKAGES = ["ultralytics", "pycocotools"]


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def pin_or_install_packages():
    """First session: install latest and record versions. Later sessions: install the
    recorded versions so every run uses identical library code."""
    os.makedirs(C.ENV_DIR, exist_ok=True)
    pin_path = os.path.join(C.ENV_DIR, "pinned_versions.json")
    pins = load_json(pin_path)
    if pins is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + PINNED_PACKAGES, check=True)
        pins = {p: _pkg_version(p) for p in PINNED_PACKAGES}
        for p in ("torch", "torchvision", "numpy"):
            pins[p] = _pkg_version(p)
        save_json(pins, pin_path)
        print("Recorded pinned versions:", pins)
    else:
        specs = [f"{p}=={pins[p]}" for p in PINNED_PACKAGES if pins.get(p)]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + specs, check=True)
        for p in ("torch", "torchvision"):
            cur = _pkg_version(p)
            if pins.get(p) and cur != pins[p]:
                print(f"WARNING: {p} is {cur} in this Colab image but runs so far used {pins[p]}. "
                      f"Report this in the appendix (or pip install {p}=={pins[p]}).")
        print("Using pinned versions:", pins)
    return pins


def gpu_info():
    info = {}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["cuda_version_torch"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
    except ImportError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True)
        info["nvidia_smi"] = out.stdout.strip()
    except Exception:
        pass
    return info


def environment_snapshot():
    snap = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        **gpu_info(),
    }
    for p in ("torch", "torchvision", "ultralytics", "pycocotools", "numpy",
              "opencv-python-headless", "opencv-python", "pillow"):
        snap[f"ver_{p}"] = _pkg_version(p)
    try:
        import psutil
        snap["ram_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    return snap


def save_pip_freeze(path):
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    with open(path, "w") as f:
        f.write(out.stdout)


def write_run_info(model_name, seed, extra=None):
    """results/<model>/seed<k>/logs/run_info.json : everything needed for the appendix."""
    logs = C.run_dir(model_name, seed, "logs")
    info = {
        "model": model_name,
        "seed": seed,
        "split_seed": C.SPLIT_SEED,
        "split_hashes": load_json(os.path.join(C.SPLITS_DIR, "manifest_hashes.json")),
        "classes": C.CLASSES,
        "excluded_prefixes": C.EXCLUDE_PREFIXES,
        "shared_protocol": dict(
            epochs=C.EPOCHS, effective_batch=C.EFFECTIVE_BATCH, optimizer=C.OPTIMIZER,
            momentum=C.MOMENTUM, nesterov=C.NESTEROV, weight_decay=C.WEIGHT_DECAY, warmup_epochs=C.WARMUP_EPOCHS,
            min_warmup_iters=C.MIN_WARMUP_ITERS, lr_schedule="linear warmup + cosine",
            final_lr_frac=C.FINAL_LR_FRAC, grad_clip=C.GRAD_CLIP_NORM, amp=C.AMP,
            ema_decay=C.EMA_DECAY, ema_tau=C.EMA_TAU, augmentation=C.AUG,
            inference=C.INFER, selection_metric=C.SELECTION_METRIC + " on val (fair_eval)",
        ),
        "model_cfg": C.MODEL_CFG[model_name],
        "environment": environment_snapshot(),
    }
    if extra:
        info.update(extra)
    save_json(info, os.path.join(logs, "run_info.json"))
    save_pip_freeze(os.path.join(C.ENV_DIR, f"pip_freeze_{model_name}_seed{seed}.txt"))
    return info


def mount_drive():
    try:
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
    except ImportError:
        print("Not running in Colab; skipping Drive mount.")


def check_gpu_consistency():
    """All runs and the speed benchmark should use the same GPU type. The first session records it."""
    lock = os.path.join(C.ENV_DIR, "gpu_lock.json")
    cur = gpu_info().get("gpu_name")
    rec = load_json(lock)
    if rec is None:
        save_json({"gpu_name": cur}, lock)
        print(f"GPU recorded for all runs: {cur}")
    elif rec["gpu_name"] != cur:
        print(f"WARNING: this session has {cur}, but earlier runs used {rec['gpu_name']}. "
              f"Change runtime type to match, or document the difference in the appendix.")
    else:
        print(f"GPU OK (same as earlier runs): {cur}")
    return cur
