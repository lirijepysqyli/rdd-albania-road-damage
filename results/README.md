# Results

Copy these from `MyDrive/RDD_fair/results/` on Google Drive before publishing:

| Folder | What to copy |
|---|---|
| `00_dataset/` | split files and manifests with SHA-256 hashes (needed to rebuild the exact split) |
| `00_environment/` | `pinned_versions.json`, pip freeze, GPU info |
| `<model>/seed0/logs/` | `train_log.csv`, `run_info.json`, `selection.json` |
| `<model>/seed0/metrics/` | metric JSON/CSV files and `operating_threshold.json` |
| `<model>/seed0/predictions/` | `val.json`, `test.json`, `albania.json` (COCO-format detections) |
| `comparison/` | cross-model tables, bootstrap results, `speed_benchmark.csv`, `reproducibility_appendix.md` |

Do **not** copy `checkpoints/` here. The selected weights (`best.pt`, one per model) are
attached to the GitHub Release instead, because GitHub does not accept files over 100 MB.
