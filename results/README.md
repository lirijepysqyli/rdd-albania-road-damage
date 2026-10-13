# Results

This folder contains the fixed data split (`00_dataset/`, with SHA-256 hashes) and the
software environment of the runs (`00_environment/`).

All other outputs from the paper's runs are attached to the
[v1.0 release](../../releases/tag/v1.0) as `results_full.zip`:

| Folder | Contents |
|---|---|
| `<model>/seed0/logs/` | training logs, selected epoch |
| `<model>/seed0/metrics/` | per-class metrics, bootstrap CIs, operating thresholds |
| `<model>/seed0/predictions/` | detections on val, test and Albania (COCO format) |
| `<model>/seed0/figures/` | PR curves, confusion matrices, prediction images |
| `comparison/` | cross-model tables and paired bootstrap tests |
| `size_and_timing/` | AP by object size and inference-time results (notebook 05) |

The trained weights (`*_best.pt`, one per model) are attached to the same release.
