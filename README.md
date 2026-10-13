# Fair comparison: SSD300-VGG16 vs Faster R-CNN R50-FPN vs YOLOv8m on RDD2022 and Albanian roads

Code for the paper *Cross-Country Evaluation of Deep Learning Detectors for Road Damage Detection
Using RDD2022 and Albanian Road Images* and the MSc thesis *Comparative Analysis of Deep Learning
Models for Road Damage Detection: A Cross-Country Evaluation on RDD2022 and Albanian Road Images*.
Lirije Pysqyli, University of New York Tirana, 2026.

All three detectors are trained and evaluated under one protocol: the same fixed RDD2022 split,
training budget, optimizer, augmentation policy and a single shared evaluator. They are then tested
on 200 held-out Albanian road images that were never used for training, model selection or
threshold tuning.

## Repository contents

```
├── notebooks/
│   ├── 01_train_SSD300_VGG16.ipynb
│   ├── 02_train_FasterRCNN_ResNet50.ipynb
│   ├── 03_train_YOLOv8m.ipynb
│   ├── 04_evaluate_and_compare.ipynb
│   ├── 05_size_and_timing.ipynb
│   └── 06_paired_bootstrap_10k.ipynb
├── rdd_fair/              shared Python package used by all notebooks
│   ├── config.py          paths and settings (edit only the top if your paths differ)
│   ├── prepare_data.py    builds the fixed RDD2022 split (seed 42) + SHA-256 hashes
│   ├── fair_eval.py       shared evaluator: P, R, F1, mAP@0.5, mAP@0.5:0.95, bootstrap, timing
│   ├── tv_models.py       torchvision SSD300-VGG16 and Faster R-CNN definitions
│   ├── tv_train.py        training loop for the torchvision models
│   ├── yolo_train.py      YOLOv8m training (Ultralytics) with the common recipe
│   ├── report.py          result tables and figures
│   └── utils.py           helpers
├── results/               split files, logs, metrics and predictions from the paper's runs
├── requirements.txt
└── LICENSE
```

The trained weights (`best.pt` for each model) are attached to the
[GitHub Release](../../releases), not stored in the repository.

## Data

- **RDD2022**: public (Arya et al., 2024), used through the Kaggle repackaging
  [aliabdelmenam/rdd-2022](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022).
- **Albanian test set**: 200 images with 753 bounding boxes, openly available on Zenodo under
  CC BY 4.0: https://doi.org/10.5281/zenodo.23351414

## Setup (Google Colab, once)

The code is written for Google Colab with Google Drive as storage.

1. Upload this repository to Google Drive as `MyDrive/RDD_fair/code/`.
2. Put the Kaggle zip at `MyDrive/RDD_SPLIT.zip`.
3. Put the Albanian test set (COCO JSON export is easiest; a YOLO export also works) in
   `MyDrive/albania_test/`. Class names must be D00, D10, D20, D40 (case does not matter).
4. If your paths differ, edit only the top of `rdd_fair/config.py`.
5. In Colab choose Runtime → Change runtime type → A100, and use the same GPU for every notebook.

The notebooks install and pin the library versions themselves. `requirements.txt` lists the versions
used for the paper, for running outside Colab.

## Run order

| Notebook | What it does |
|---|---|
| 01_train_SSD300_VGG16.ipynb | 1 seed × 50 epochs |
| 02_train_FasterRCNN_ResNet50.ipynb | 1 seed × 50 epochs |
| 03_train_YOLOv8m.ipynb | 1 seed × 50 epochs |
| 04_evaluate_and_compare.ipynb | predictions, metrics, bootstrap CIs, paired tests, error analysis, Albania results |
| 05_size_and_timing.ipynb | AP by object size (small/medium/large) and the inference-time measurements reported in the paper |
| 06_paired_bootstrap_10k.ipynb | 95% confidence intervals and paired two-sided p-values reported in the paper (10,000 resamples) |

The first notebook you run creates the fixed split (seed 42) and saves it to Drive. Every later
session rebuilds the identical dataset from it and checks the SHA-256 hashes. If Colab disconnects,
reconnect and *Run all* again: training resumes from the last saved epoch.

The code supports several seeds (`SEEDS` in `config.py`). The paper uses one seed per model (seed 0).
Notebooks 01 and 03 end with an optional cell that measures peak GPU memory during training (Table 5);
it is not needed for the main results.

## Outputs (all in one Drive folder)

```
MyDrive/RDD_fair/results/
├── 00_dataset/            split JSONs + manifests (+SHA-256), class and country counts
├── 00_environment/        pinned versions, pip freeze per run, GPU info
├── ssd300_vgg16/seed0/
│   ├── checkpoints/       last.pt (every epoch, resume) · best.pt (selected epoch)
│   ├── logs/              train_log.csv · run_info.json · selection.json · DONE
│   ├── predictions/       val.json · test.json · albania.json
│   ├── metrics/           per-class AP/P/R/F1, bootstrap CIs, confusion matrix, error breakdown
│   └── figures/           PR curves, confusion matrix, error galleries, Albania predictions
├── fasterrcnn_r50_fpn_v2/seed0/   (same layout)
├── yolov8m/seed0/                 (same layout; Ultralytics files in seed0/ultralytics/)
└── comparison/            cross-model tables (.csv/.md/.tex), paired bootstrap tests,
                           D20/D40 analysis, Albania grids, speed_benchmark.csv,
                           training_curves_val.png, reproducibility_appendix.md
```

## Protocol (identical for all models unless stated)

| Item | Setting |
|---|---|
| Data | RDD2022, 4 classes (D00, D10, D20, D40); other labels dropped; China_Drone excluded |
| Split | 70/15/15 stratified by country × rarest class, seed 42: 25,187 / 5,396 / 5,401 images |
| Albania | 200 images, 753 boxes, test only (never used for training, selection or thresholds) |
| Pretraining | COCO detection weights (torchvision COCO_V1, Ultralytics yolov8m.pt) |
| Input | SSD 300×300 (native); YOLOv8m 640×640 (letterboxed); Faster R-CNN longest side 640 px, aspect ratio kept |
| Classes in the head | torchvision models: 4 damage classes + background = 5 outputs; YOLOv8m: 4 |
| Epochs / batch | 50 / effective 16 (Faster R-CNN: 8 × 2 gradient accumulation) |
| Seeds | 1 per model |
| Optimizer | SGD, Nesterov momentum 0.9, wd 5e-4 (no decay on bias/norm), grad clip 10, AMP, EMA 0.9999 |
| LR | base: SSD 0.002 · Faster R-CNN 0.02 · YOLOv8m 0.01; 3-epoch warm-up, cosine to 1% |
| Augmentation | h-flip 0.5, HSV value ±0.2, scale ±0.1; no mosaic/mixup/translate/rotation |
| Model selection | best validation mAP@0.5:0.95 from the shared evaluator |
| Inference | conf ≥ 0.001 for AP, class-wise NMS IoU 0.5, max 100 detections |
| P/R/F1 | per-model threshold maximising macro-F1 on validation, applied unchanged to test and Albania |
| Statistics | notebook 06: image-level bootstrap, 10,000 resamples (seed 12345); 95% CIs and paired two-sided p-values, both models scored on the same resampled images (notebook 04 uses a quicker 1,000-resample version during evaluation) |
| Speed | notebook 05: batch 1, FP32, one NVIDIA A100-SXM4-40GB, 550 randomly sampled RDD2022 test images (seed 0), first 10 as warm-up and 540 timed; includes pre-processing, forward pass and NMS, not image loading |

## Citation

If you use this code, please cite the paper (reference to be added on publication) and the
Albanian test set:

> Pysqyli, L.: Albanian road damage test set (v1.0). Zenodo (2026). https://doi.org/10.5281/zenodo.23351414

and the code:

> Pysqyli, L.: rdd-albania-road-damage (v1.0). Zenodo (2026). https://doi.org/10.5281/zenodo.23352926

## License

Code: MIT (see `LICENSE`). Albanian test set: CC BY 4.0. RDD2022 follows its own licence.
