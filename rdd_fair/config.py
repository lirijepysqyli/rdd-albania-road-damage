"""
Single source of truth for the fair SSD / Faster R-CNN / YOLOv8 comparison.
Every notebook imports from here, so changing a value here changes it for all
three models. Values that intentionally differ per model are in MODEL_CFG and
are documented in the reproducibility appendix.
"""
import os

# --------------------------------------------------------------------------
# Paths
# Everything persistent is saved under ONE Drive folder: RDD_fair/results/
#
#   results/
#     00_dataset/         split manifests (+ SHA-256), class/country counts
#     00_environment/     pinned package versions, pip freeze, GPU info
#     ssd300_vgg16/
#       seed0/            (the paper uses one seed per model; add more to SEEDS if needed)
#         checkpoints/    last.pt (every epoch, for resume), best.pt (selected)
#         logs/           train_log.csv, run_info.json, DONE marker
#         predictions/    val/test/albania detections (COCO json)
#         metrics/        per-class AP/P/R/F1, confusion matrix, error breakdown
#         figures/        PR curves, confusion matrix, FP/FN galleries
#       summary/          mean +- std over seeds (just the value when there is one seed)
#     fasterrcnn_r50_fpn_v2/   (same layout)
#     yolov8m/                 (same layout; Ultralytics' own files in seedN/ultralytics/)
#     comparison/         cross-model tables, Albania side-by-side grids, speed, error analysis
#
# /content/... is fast local disk (wiped each Colab session) used only for raw data.
# --------------------------------------------------------------------------
DRIVE_ROOT   = "/content/drive/MyDrive/RDD_fair"
RDD_ZIP      = "/content/drive/MyDrive/RDD_SPLIT.zip"     # Kaggle aliabdelmenam/rdd-2022 zip
ALBANIA_SRC  = "/content/drive/MyDrive/albania_test"      # Roboflow export (COCO json or YOLO folder)

LOCAL_ROOT   = "/content/rdd_local"                       # unzipped raw data (local, fast)
DATA_ROOT    = "/content/rdd_fair_data"                   # canonical dataset rebuilt from manifests

RESULTS_DIR    = os.path.join(DRIVE_ROOT, "results")
SPLITS_DIR     = os.path.join(RESULTS_DIR, "00_dataset")
ENV_DIR        = os.path.join(RESULTS_DIR, "00_environment")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "comparison")

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
CLASSES = ["D00", "D10", "D20", "D40"]          # COCO category_id = index + 1
CLASS_ALIASES = {                                         # name normalisation (case-insensitive)
    "d00": "D00", "d10": "D10", "d20": "D20", "d40": "D40",
}
YOLO_CLASS_NAMES = ["D00", "D10", "D20", "excluded_marking_blur", "D40"]   # Kaggle class numbers 0-4
EXCLUDE_PREFIXES = ["China_Drone"]                        # aerial view; excluded by design
SPLIT_SEED   = 42                                         # split never changes between runs
SPLIT_RATIOS = (0.70, 0.15, 0.15)                         # train / val / test
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".bmp")

# --------------------------------------------------------------------------
# Shared training protocol (identical for all models)
# --------------------------------------------------------------------------
SEEDS            = [0]
EPOCHS           = 50
EFFECTIVE_BATCH  = 16
OPTIMIZER        = "SGD"
MOMENTUM         = 0.9
NESTEROV         = True          # Ultralytics SGD uses Nesterov; applied to all
WEIGHT_DECAY     = 5e-4          # not applied to biases / norm layers (same as Ultralytics)
WARMUP_EPOCHS    = 3             # linear warmup, at least 100 iterations
MIN_WARMUP_ITERS = 100
FINAL_LR_FRAC    = 0.01          # cosine decay to 1% of base LR
GRAD_CLIP_NORM   = 10.0          # Ultralytics default, applied to all
AMP              = True
EMA_DECAY        = 0.9999        # Ultralytics ModelEMA formula, applied to all
EMA_TAU          = 2000
NUM_WORKERS      = 12

# Shared augmentation policy (Ultralytics extras switched off)
AUG = dict(
    hflip_p    = 0.5,
    brightness = 0.2,    # HSV value gain ~ U(0.8, 1.2)  (== Ultralytics hsv_v=0.2)
    scale      = 0.1,    # random scale ~ U(0.9, 1.1) about image centre, fill 114
)

# --------------------------------------------------------------------------
# Shared inference / evaluation protocol
# --------------------------------------------------------------------------
INFER = dict(
    conf_for_map = 0.001,   # low threshold for AP computation
    nms_iou      = 0.5,     # class-wise NMS
    max_det      = 100,
)
SELECTION_METRIC = "mAP50_95"   # on val, computed by fair_eval; best epoch -> best.pt
F1_IOU           = 0.5          # IoU used for P/R/F1, confusion matrix, error analysis
BOOTSTRAP_B      = 1000
BOOTSTRAP_SEED   = 12345
SPEED_WARMUP     = 50
SPEED_N_IMAGES   = 500

# --------------------------------------------------------------------------
# Per-model settings (only what must differ for architectural reasons)
# --------------------------------------------------------------------------
MODEL_CFG = {
    "ssd300_vgg16": dict(
        family="torchvision", imgsz=300, base_lr=0.002, batch=16,
        weights="SSD300_VGG16_Weights.COCO_V1", trainable_backbone_layers=5,
    ),
    "fasterrcnn_r50_fpn_v2": dict(
        family="torchvision", imgsz=640, base_lr=0.02, batch=8,   # x2 accumulation = 16, gradient checkpointing enabled
        weights="FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1", trainable_backbone_layers=5,
    ),
    "yolov8m": dict(
        family="ultralytics", imgsz=640, base_lr=0.01, batch=16,
        weights="yolov8m.pt",
    ),
}
MODEL_LABELS = {"ssd300_vgg16": "SSD300-VGG16",
                "fasterrcnn_r50_fpn_v2": "Faster R-CNN R50-FPN",
                "yolov8m": "YOLOv8m"}


RUN_SUBDIRS = ("checkpoints", "logs", "predictions", "metrics", "figures")


def model_dir(model_name):
    return os.path.join(RESULTS_DIR, model_name)


def run_dir(model_name, seed, sub=None):
    """results/<model>/seed<k>[/<sub>] ; created on first use."""
    d = os.path.join(RESULTS_DIR, model_name, f"seed{seed}")
    if sub:
        d = os.path.join(d, sub)
    os.makedirs(d, exist_ok=True)
    return d


def summary_dir(model_name):
    d = os.path.join(RESULTS_DIR, model_name, "summary")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_dirs():
    for d in (RESULTS_DIR, SPLITS_DIR, ENV_DIR, COMPARISON_DIR):
        os.makedirs(d, exist_ok=True)
    for m in MODEL_CFG:
        for k in SEEDS:
            for sub in RUN_SUBDIRS:
                run_dir(m, k, sub)
        summary_dir(m)
