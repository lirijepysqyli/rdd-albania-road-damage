"""
Everything after training. All outputs go to Drive:

  results/<model>/seed<k>/predictions/{val,test,albania}.json
  results/<model>/seed<k>/metrics/{test,albania}_metrics.json + CSV tables
  results/<model>/seed<k>/figures/...            PR curves, confusion matrices, errors, galleries,
                                                 albania_predictions/ (every Albania image annotated)
  results/<model>/summary/                       mean +- std over seeds
  results/comparison/                            cross-model tables, paired bootstrap, D20/D40
                                                 error analysis, Albania side-by-side grids, speed,
                                                 reproducibility_appendix.md
"""
import collections
import json
import os
import time
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import prepare_data as P
from .fair_eval import ERROR_TYPES, IOU_THRS_COCO, IOU_THRS_SWEEP, Evaluator, full_evaluation
from .utils import environment_snapshot, load_json, save_json

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="No artists with labels")
MODELS = list(C.MODEL_CFG)
FOCUS = [c for c in ("D20", "D40", "Repair") if c in C.CLASSES]
COLORS = {"D00": (230, 25, 75), "D10": (60, 180, 75), "D20": (255, 225, 25),
          "D40": (0, 130, 200), "Repair": (245, 130, 48)}   # RGB


# ==========================================================================
# predictions
# ==========================================================================
def load_model(model_name, seed, device="cuda"):
    ck = os.path.join(C.run_dir(model_name, seed, "checkpoints"), "best.pt")
    if C.MODEL_CFG[model_name]["family"] == "ultralytics":
        from ultralytics import YOLO
        return YOLO(ck)
    from .tv_models import load_for_inference
    return load_for_inference(model_name, ck, device)


def predict_split(model_name, model, split):
    if C.MODEL_CFG[model_name]["family"] == "ultralytics":
        from .yolo_train import predict
    else:
        from .tv_models import predict
    return predict(model, P.ann_path(split), P.img_dir(split))


def run_predictions(model_name, seed, splits=("val", "test", "albania"), overwrite=False):
    pred_dir = C.run_dir(model_name, seed, "predictions")
    todo = [s for s in splits if overwrite or not os.path.exists(os.path.join(pred_dir, f"{s}.json"))]
    if not todo:
        return
    if not os.path.exists(os.path.join(C.run_dir(model_name, seed, "logs"), "DONE")):
        print(f"WARNING: {model_name} seed {seed} has not finished training; skipping predictions.")
        return
    model = load_model(model_name, seed)
    for s in todo:
        t = time.time()
        dets = predict_split(model_name, model, s)
        save_json(dets, os.path.join(pred_dir, f"{s}.json"))
        print(f"  {model_name} seed{seed} {s}: {len(dets)} dets ({time.time() - t:.0f}s)")
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def load_dets(model_name, seed, split):
    return load_json(os.path.join(C.run_dir(model_name, seed, "predictions"), f"{split}.json"))


# ==========================================================================
# per-run evaluation
# ==========================================================================
_EVAL_CACHE = {}


def evaluator(split):
    if split not in _EVAL_CACHE:
        _EVAL_CACHE[split] = Evaluator(P.ann_path(split))
    return _EVAL_CACHE[split]


def evaluate_run(model_name, seed, eval_splits=("test", "albania"), overwrite=False):
    """Operating threshold from VAL (max macro-F1), then full evaluation on test & Albania."""
    met_dir = C.run_dir(model_name, seed, "metrics")
    fig_dir = C.run_dir(model_name, seed, "figures")
    thr_path = os.path.join(met_dir, "operating_threshold.json")
    if os.path.exists(thr_path) and not overwrite:
        thr = load_json(thr_path)["threshold"]
    else:
        ev = evaluator("val")
        tables = ev.match_tables(load_dets(model_name, seed, "val"))
        thr, f1, curve = ev.choose_threshold(tables)
        save_json({"threshold": thr, "val_macro_F1": f1, "rule": "max macro-F1 on val @IoU0.5",
                   "curve": curve}, thr_path)
    out = {}
    for split in eval_splits:
        p = os.path.join(met_dir, f"{split}_metrics.json")
        if os.path.exists(p) and not overwrite:
            out[split] = load_json(p)
            continue
        dets = load_dets(model_name, seed, split)
        ev = evaluator(split)
        res, tables = full_evaluation(ev, dets, thr)
        gt_rec, fp_rec = res["analysis"].pop("gt_records"), res["analysis"].pop("fp_records")
        pd.DataFrame(gt_rec).to_csv(os.path.join(met_dir, f"{split}_gt_records.csv"), index=False)
        pd.DataFrame(fp_rec).to_csv(os.path.join(met_dir, f"{split}_fp_records.csv"), index=False)
        save_json(res, p)
        _per_run_tables(res, met_dir, split)
        _per_run_figures(ev, tables, res, fig_dir, split, model_name, seed)
        out[split] = res
        if not res["own_ap_check"]["agrees_with_pycocotools"]:
            print("WARNING: own AP differs from pycocotools by >0.5 points; check inputs.")
    return out


def _per_run_tables(res, met_dir, split):
    rows = []
    for c in C.CLASSES:
        pc, pr = res["coco"]["per_class"].get(c, {}), res["prf1"].get(c, {})
        rows.append({"class": c, "n_gt": pr.get("n_gt"), "AP50": pc.get("AP50"), "AP50_95": pc.get("AP50_95"),
                     "AP_small": pc.get("AP_small"), "AP_medium": pc.get("AP_medium"), "AP_large": pc.get("AP_large"),
                     "precision": pr.get("precision"), "recall": pr.get("recall"), "F1": pr.get("F1"),
                     "TP": pr.get("TP"), "FP": pr.get("FP"), "FN": pr.get("FN")})
    pd.DataFrame(rows).round(2).to_csv(os.path.join(met_dir, f"{split}_per_class.csv"), index=False)
    pd.DataFrame(res["analysis"]["errors"]).T.fillna(0).astype(int).to_csv(
        os.path.join(met_dir, f"{split}_error_breakdown.csv"))
    cm = pd.DataFrame(res["analysis"]["confusion_matrix"], index=res["analysis"]["cm_labels"],
                      columns=res["analysis"]["cm_labels"])
    cm.index.name = "true \\ predicted"
    cm.to_csv(os.path.join(met_dir, f"{split}_confusion_matrix.csv"))
    pd.DataFrame(res["per_country"]).T.round(2).to_csv(os.path.join(met_dir, f"{split}_per_country.csv"))
    if "bootstrap" in res:
        pd.DataFrame(res["bootstrap"]).T.round(2).to_csv(os.path.join(met_dir, f"{split}_bootstrap_ci.csv"))


def _per_run_figures(ev, tables, res, fig_dir, split, model_name, seed):
    label = f"{C.MODEL_LABELS[model_name]} (seed {seed}) - {split}"
    # PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, (r, p, _) in ev.pr_curves(tables).items():
        ax.plot(r, np.maximum.accumulate(p[::-1])[::-1], label=f"{c} AP50={res['coco']['per_class'][c]['AP50']:.1f}",
                color=np.array(COLORS[c]) / 255)
    ax.set(xlabel="Recall", ylabel="Precision", title=f"PR @IoU0.5\n{label}", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{split}_pr_curves.png"), dpi=200)
    plt.close(fig)
    # confusion matrix (row-normalised)
    cm = np.array(res["analysis"]["confusion_matrix"], float)
    labels = res["analysis"]["cm_labels"]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="True", title=f"Confusion matrix (conf>={res['threshold']:.2f}, IoU0.5)\n{label}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{split}_confusion_matrix.png"), dpi=200)
    plt.close(fig)
    # error breakdown
    err = pd.DataFrame(res["analysis"]["errors"]).T.reindex(columns=ERROR_TYPES + ["missed", "missed_but_overlapped"]).fillna(0)
    ax = err.plot(kind="bar", stacked=True, figsize=(7, 4.5), colormap="tab10")
    ax.set(title=f"Error breakdown (counts)\n{label}", ylabel="count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{split}_error_breakdown.png"), dpi=200)
    plt.close()


# ==========================================================================
# seed summaries and cross-model comparison
# ==========================================================================
def _finished_seeds(model_name):
    return [s for s in C.SEEDS if os.path.exists(os.path.join(C.run_dir(model_name, s, "metrics"), "test_metrics.json"))]


def _ms(vals):
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def fmt_mean_std(t):
    """Mean +/- std, but just the mean (no '+/- 0.0') when there's only one seed / std is 0."""
    mean, std = t
    if mean != mean:  # NaN
        return "nan"
    return f"{mean:.1f}" if not std else f"{mean:.1f} ± {std:.1f}"


def summarise_model(model_name, split="test"):
    seeds = _finished_seeds(model_name)
    runs = [load_json(os.path.join(C.run_dir(model_name, s, "metrics"), f"{split}_metrics.json")) for s in seeds]
    if not runs:
        return None
    overall = {}
    for k in ("mAP50", "mAP50_95", "mAP75", "AP_small", "AP_medium", "AP_large"):
        overall[k] = _ms([r["coco"][k] for r in runs])
    overall["macro_F1"] = _ms([r["prf1"]["macro"]["F1"] for r in runs])
    overall["macro_P"] = _ms([r["prf1"]["macro"]["precision"] for r in runs])
    overall["macro_R"] = _ms([r["prf1"]["macro"]["recall"] for r in runs])
    per_class = {}
    for c in C.CLASSES:
        per_class[c] = {k: _ms([r["coco"]["per_class"][c][k] for r in runs]) for k in ("AP50", "AP50_95")}
        per_class[c].update({k: _ms([r["prf1"][c][k] for r in runs]) for k in ("precision", "recall", "F1")})
    sel = [load_json(os.path.join(C.run_dir(model_name, s, "logs"), "selection.json"), {}) for s in seeds]
    thr = [load_json(os.path.join(C.run_dir(model_name, s, "metrics"), "operating_threshold.json"))["threshold"] for s in seeds]
    summ = {"seeds": seeds, "overall": overall, "per_class": per_class,
            "best_epochs": [x.get("best_epoch") for x in sel], "thresholds": thr}
    out = C.summary_dir(model_name)
    save_json(summ, os.path.join(out, f"{split}_summary.json"))
    pd.DataFrame({c: {k: fmt_mean_std(v) for k, v in d.items()} for c, d in per_class.items()}).T.to_csv(
        os.path.join(out, f"{split}_per_class_mean_std.csv"))
    pd.Series({k: fmt_mean_std(v) for k, v in overall.items()}).to_csv(os.path.join(out, f"{split}_overall_mean_std.csv"))
    return summ


def seed_averaged_paired_bootstrap(split, B=C.BOOTSTRAP_B, seed=C.BOOTSTRAP_SEED):
    """For each model: CI of the seed-averaged metric (image resampling). For each model pair: CI of
    the paired difference (same resampled images for both). Captures test-set sampling uncertainty;
    seed std captures training randomness."""
    ev = evaluator(split)
    runs = {}
    for m in MODELS:
        seeds = _finished_seeds(m)
        if not seeds:
            continue
        runs[m] = []
        for s in seeds:
            thr = load_json(os.path.join(C.run_dir(m, s, "metrics"), "operating_threshold.json"))["threshold"]
            runs[m].append((ev.match_tables(load_dets(m, s, split), iou_thrs=IOU_THRS_COCO), thr))
    rng = np.random.default_rng(seed)
    n = len(ev.img_ids)

    def metrics(tables, thr, w):
        ap = np.array(list(ev.ap_from_tables(tables, w).values()))
        f = ev.pr_f1(tables, thr, img_weights=w)
        d = {"mAP50": np.nanmean(ap[:, 0]), "mAP50_95": np.nanmean(ap), "macro_F1": f["macro"]["F1"]}
        for k, c in enumerate(C.CLASSES):
            d[f"{c}_AP50"] = ap[k, 0]
            d[f"{c}_F1"] = f[c]["F1"]
        return d

    def avg(m, w):
        ds = [metrics(t, th, w) for t, th in runs[m]]
        return {k: float(np.nanmean([d[k] for d in ds])) for k in ds[0]}

    point = {m: avg(m, None) for m in runs}
    samples = {m: collections.defaultdict(list) for m in runs}
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for m in runs:
            for k, v in avg(m, w).items():
                samples[m][k].append(v)
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{B}")
    ci_rows, diff_rows = [], []
    for m in runs:
        for k in point[m]:
            v = np.asarray(samples[m][k])
            v = v[~np.isnan(v)]
            lo, hi = np.percentile(v, [2.5, 97.5]) if v.size else (np.nan, np.nan)
            ci_rows.append({"model": C.MODEL_LABELS[m], "metric": k, "point": point[m][k], "ci_low": lo, "ci_high": hi})
    names = list(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b_ = names[i], names[j]
            for k in point[a]:
                d = np.asarray(samples[a][k]) - np.asarray(samples[b_][k])
                d = d[~np.isnan(d)]
                lo, hi = np.percentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan)
                p = 2 * min((d <= 0).mean(), (d >= 0).mean()) if d.size else np.nan
                diff_rows.append({"A": C.MODEL_LABELS[a], "B": C.MODEL_LABELS[b_], "metric": k,
                                  "A_minus_B": point[a][k] - point[b_][k], "ci_low": lo, "ci_high": hi,
                                  "bootstrap_p": p, "significant_95": bool(lo > 0 or hi < 0)})
    ci = pd.DataFrame(ci_rows).round(2)
    diff = pd.DataFrame(diff_rows).round(3)
    ci.to_csv(os.path.join(C.COMPARISON_DIR, f"{split}_seed_averaged_bootstrap_ci.csv"), index=False)
    diff.to_csv(os.path.join(C.COMPARISON_DIR, f"{split}_paired_differences.csv"), index=False)
    return ci, diff


def compare_models(split="test"):
    summ = {m: summarise_model(m, split) for m in MODELS}
    summ = {m: s for m, s in summ.items() if s}
    overall = pd.DataFrame({C.MODEL_LABELS[m]: {k: fmt_mean_std(v) for k, v in s["overall"].items()} for m, s in summ.items()}).T
    overall["seeds"] = [len(s["seeds"]) for s in summ.values()]
    overall["best_epochs"] = [s["best_epochs"] for s in summ.values()]
    overall.to_csv(os.path.join(C.COMPARISON_DIR, f"{split}_overall.csv"))
    tabs = {}
    for metric in ("AP50", "AP50_95", "precision", "recall", "F1"):
        t = pd.DataFrame({C.MODEL_LABELS[m]: {c: fmt_mean_std(s["per_class"][c][metric]) for c in C.CLASSES}
                          for m, s in summ.items()})
        t.to_csv(os.path.join(C.COMPARISON_DIR, f"{split}_per_class_{metric}.csv"))
        tabs[metric] = t
    with open(os.path.join(C.COMPARISON_DIR, f"{split}_tables.md"), "w") as f:
        f.write(f"# {split}: mean ± std over seeds\n\n## Overall\n\n{overall.to_markdown()}\n\n")
        for k, t in tabs.items():
            f.write(f"## Per-class {k}\n\n{t.to_markdown()}\n\n")
    with open(os.path.join(C.COMPARISON_DIR, f"{split}_tables.tex"), "w") as f:
        f.write(overall.to_latex())
        for t in tabs.values():
            f.write(t.to_latex())
    return overall, tabs


def focus_error_analysis(split="test"):
    """D20 / D40: error types, confusions, AP vs IoU, recall by size; mean over seeds."""
    rows, conf_rows, size_rows, iou_rows = [], [], [], []
    for m in MODELS:
        for s in _finished_seeds(m):
            r = load_json(os.path.join(C.run_dir(m, s, "metrics"), f"{split}_metrics.json"))
            for c in FOCUS:
                e = r["analysis"]["errors"].get(c, {})
                n_fp = sum(e.get(t, 0) for t in ERROR_TYPES)
                rows.append({"model": C.MODEL_LABELS[m], "seed": s, "class": c,
                             "n_gt": r["prf1"][c]["n_gt"], "TP": r["prf1"][c]["TP"], "FP": n_fp,
                             **{f"FP_{t}": e.get(t, 0) for t in ERROR_TYPES},
                             "FN_missed": e.get("missed", 0), "FN_overlapped": e.get("missed_but_overlapped", 0)})
                for other, n in r["analysis"]["classification_confused_with"].get(c, {}).items():
                    conf_rows.append({"model": C.MODEL_LABELS[m], "seed": s, "predicted": c, "true": other, "n": n})
                for sz, v in r["analysis"]["size_recall"][c].items():
                    size_rows.append({"model": C.MODEL_LABELS[m], "seed": s, "class": c, "size": sz, **v})
                for t, ap in r["ap_vs_iou"][c].items():
                    iou_rows.append({"model": C.MODEL_LABELS[m], "seed": s, "class": c, "iou": float(t), "AP": ap})
    if not rows:
        return
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("model", "seed", "class")]
    df.groupby(["class", "model"])[num].mean().round(1).to_csv(
        os.path.join(C.COMPARISON_DIR, f"{split}_focus_errors_mean_over_seeds.csv"))
    pd.DataFrame(conf_rows).groupby(["predicted", "true", "model"])["n"].mean().round(1).to_csv(
        os.path.join(C.COMPARISON_DIR, f"{split}_focus_confusions.csv"))
    sdf = pd.DataFrame(size_rows).groupby(["class", "size", "model"])[["n_gt", "recall"]].mean().round(1)
    sdf.to_csv(os.path.join(C.COMPARISON_DIR, f"{split}_focus_recall_by_size.csv"))
    idf = pd.DataFrame(iou_rows).groupby(["class", "model", "iou"])["AP"].agg(["mean", "std"]).reset_index()
    idf.round(2).to_csv(os.path.join(C.COMPARISON_DIR, f"{split}_focus_ap_vs_iou.csv"), index=False)

    # figure 1: AP vs IoU threshold
    fig, axes = plt.subplots(1, len(FOCUS), figsize=(15, 4), sharey=True)
    for ax, c in zip(axes, FOCUS):
        for m in idf["model"].unique():
            d = idf[(idf["class"] == c) & (idf["model"] == m)]
            ax.errorbar(d["iou"], d["mean"], yerr=d["std"], marker="o", capsize=3, label=m)
        ax.set(title=f"{c}: AP vs IoU threshold", xlabel="IoU threshold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AP (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.COMPARISON_DIR, f"{split}_focus_ap_vs_iou.png"), dpi=200)
    plt.close(fig)
    # figure 2: FP/FN composition
    mean = df.groupby(["class", "model"])[[f"FP_{t}" for t in ERROR_TYPES] + ["FN_missed", "FN_overlapped"]].mean()
    fig, axes = plt.subplots(1, len(FOCUS), figsize=(16, 4.5))
    for ax, c in zip(axes, FOCUS):
        if c in mean.index.get_level_values(0):
            mean.loc[c].plot(kind="barh", stacked=True, ax=ax, legend=(c == FOCUS[-1]), colormap="tab10")
        ax.set(title=f"{c}: false positives / false negatives", xlabel="count (mean over seeds)")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.COMPARISON_DIR, f"{split}_focus_error_composition.png"), dpi=200)
    plt.close(fig)
    # figure 3: recall by size
    sdf2 = sdf.reset_index()
    fig, axes = plt.subplots(1, len(FOCUS), figsize=(15, 4), sharey=True)
    for ax, c in zip(axes, FOCUS):
        d = sdf2[sdf2["class"] == c].pivot(index="size", columns="model", values="recall").reindex(["small", "medium", "large"])
        d.plot(kind="bar", ax=ax, rot=0)
        ns = sdf2[sdf2["class"] == c].groupby("size")["n_gt"].mean().reindex(["small", "medium", "large"])
        ax.set(title=f"{c}: recall by size (n_gt " + "/".join(str(int(x)) if x == x else "0" for x in ns) + ")",
               ylabel="recall (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.COMPARISON_DIR, f"{split}_focus_recall_by_size.png"), dpi=200)
    plt.close(fig)


def training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for m in MODELS:
        for s in C.SEEDS:
            p = os.path.join(C.run_dir(m, s, "logs"), "train_log.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p).drop_duplicates("epoch", keep="last")
            kw = dict(label=f"{C.MODEL_LABELS[m]} s{s}", alpha=0.85)
            axes[0].plot(d["epoch"], d["val_mAP50"], **kw)
            axes[1].plot(d["epoch"], d["val_mAP50_95"], **kw)
    for ax, t in zip(axes, ["val mAP@0.5", "val mAP@0.5:0.95"]):
        ax.set(xlabel="epoch", ylabel="%", title=t)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.COMPARISON_DIR, "training_curves_val.png"), dpi=200)
    plt.close(fig)


# ==========================================================================
# visualisation
# ==========================================================================
def _draw(img, boxes, title=None, thick=None):
    img = img.copy()
    k = max(img.shape[:2]) / 640.0                     # scale lines/text with image size
    thick = thick or max(2, int(round(2 * k)))
    fs, ft = 0.5 * k, max(1, int(round(k)))
    for b in boxes:
        x, y, w, h = map(int, b["bbox"])
        c = COLORS[b["cls"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), c, thick)
        txt = b["cls"] + (f" {b['score']:.2f}" if "score" in b else "")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        cv2.rectangle(img, (x, max(0, y - th - 4)), (x + tw + 2, y), c, -1)
        cv2.putText(img, txt, (x + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft, cv2.LINE_AA)
    return img


def _gt_boxes(ev, iid):
    out = []
    for cid in ev.cat_ids:
        for b in ev.gt_boxes[(iid, cid)]:
            out.append({"bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "cls": ev.cat_names[cid]})
    return out


def _pred_boxes(dets, iid, thr):
    return [{"bbox": d["bbox"], "cls": C.CLASSES[d["category_id"] - 1], "score": d["score"]}
            for d in dets if d["image_id"] == iid and d["score"] >= thr]


def save_all_predictions(model_name, split="albania", seed=0):
    """Every image of `split` with this model's detections drawn at its operating threshold.
    Fine for Albania (200 images); for `test` (thousands of images) prefer prediction_grid()
    for a representative sample instead of calling this on the whole split."""
    ev = evaluator(split)
    dets = load_dets(model_name, seed, split)
    thr = load_json(os.path.join(C.run_dir(model_name, seed, "metrics"), "operating_threshold.json"))["threshold"]
    out = os.path.join(C.run_dir(model_name, seed, "figures"), f"{split}_predictions")
    os.makedirs(out, exist_ok=True)
    by_img = collections.defaultdict(list)
    for d in dets:
        by_img[d["image_id"]].append(d)
    for iid in ev.img_ids:
        im = ev.images[iid]
        img = cv2.cvtColor(cv2.imread(os.path.join(P.img_dir(split), im["file_name"])), cv2.COLOR_BGR2RGB)
        vis = _draw(img, _pred_boxes(by_img[iid], iid, thr))
        cv2.imwrite(os.path.join(out, im["file_name"]), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def save_all_albania_predictions(model_name, seed=0):
    """Back-compat wrapper: every Albania image with this model's detections."""
    return save_all_predictions(model_name, "albania", seed)


def prediction_grid(split="albania", seed=0, n_images=12, rows_per_page=4, pick_seed=0):
    """GT | SSD | Faster R-CNN | YOLO side by side, for ANY split (test or albania).
    Images chosen to cover every class."""
    ev = evaluator(split)
    rng = np.random.default_rng(pick_seed)
    by_class = collections.defaultdict(list)
    for iid in ev.img_ids:
        for cid in ev.cat_ids:
            if len(ev.gt_boxes[(iid, cid)]):
                by_class[ev.cat_names[cid]].append(iid)
    chosen = []
    per_class = max(1, n_images // len(C.CLASSES))
    for c in C.CLASSES:
        cand = [i for i in by_class.get(c, []) if i not in chosen]
        chosen += list(rng.choice(cand, min(per_class, len(cand)), replace=False)) if cand else []
    rest = [i for i in ev.img_ids if i not in chosen]
    if len(chosen) < n_images and rest:
        chosen += list(rng.choice(rest, min(n_images - len(chosen), len(rest)), replace=False))
    models = [m for m in MODELS if os.path.exists(os.path.join(C.run_dir(m, seed, "predictions"), f"{split}.json"))]
    dets = {m: load_dets(m, seed, split) for m in models}
    thr = {m: load_json(os.path.join(C.run_dir(m, seed, "metrics"), "operating_threshold.json"))["threshold"] for m in models}
    out_dir = os.path.join(C.COMPARISON_DIR, f"{split}_grids")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for page in range(0, len(chosen), rows_per_page):
        ids = chosen[page:page + rows_per_page]
        fig, axes = plt.subplots(len(ids), 1 + len(models), figsize=(4.2 * (1 + len(models)), 3.6 * len(ids)), squeeze=False)
        for r, iid in enumerate(ids):
            im = ev.images[int(iid)]
            img = cv2.cvtColor(cv2.imread(os.path.join(P.img_dir(split), im["file_name"])), cv2.COLOR_BGR2RGB)
            panels = [("Ground truth", _draw(img, _gt_boxes(ev, int(iid))))]
            for m in models:
                panels.append((f"{C.MODEL_LABELS[m]} (conf≥{thr[m]:.2f})", _draw(img, _pred_boxes(dets[m], int(iid), thr[m]))))
            for c, (t, p) in enumerate(panels):
                axes[r, c].imshow(p)
                axes[r, c].set_axis_off()
                if r == 0:
                    axes[r, c].set_title(t, fontsize=11)
            axes[r, 0].text(5, 20, im["file_name"][:40], fontsize=7, color="white",
                            bbox=dict(facecolor="black", alpha=0.5, pad=1))
        handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(COLORS[c]) / 255) for c in C.CLASSES]
        fig.legend(handles, C.CLASSES, loc="lower center", ncol=5)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        p = os.path.join(out_dir, f"{split}_grid_seed{seed}_page{page // rows_per_page + 1}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def albania_grid(seed=0, n_images=12, rows_per_page=4, pick_seed=0):
    """Back-compat wrapper: GT | SSD | Faster R-CNN | YOLO grid for the Albania held-out set."""
    return prediction_grid("albania", seed, n_images, rows_per_page, pick_seed)


def error_gallery(model_name, seed=0, split="test", classes=FOCUS, k=8):
    """For each focus class: top-scoring false positives and missed ground truths (crops)."""
    ev = evaluator(split)
    met = C.run_dir(model_name, seed, "metrics")
    fp = pd.read_csv(os.path.join(met, f"{split}_fp_records.csv"))
    gt = pd.read_csv(os.path.join(met, f"{split}_gt_records.csv"))
    out = os.path.join(C.run_dir(model_name, seed, "figures"), "error_galleries")
    os.makedirs(out, exist_ok=True)

    def crop(iid, xyxy, pad=0.4):
        im = ev.images[int(iid)]
        img = cv2.cvtColor(cv2.imread(os.path.join(P.img_dir(split), im["file_name"])), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        X1, Y1 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
        X2, Y2 = int(min(img.shape[1], x2 + pad * w)), int(min(img.shape[0], y2 + pad * h))
        vis = _draw(img, _gt_boxes(ev, int(iid)), thick=1)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
        return vis[Y1:Y2, X1:X2]

    for c in classes:
        f = fp[fp["category"] == c].sort_values("score", ascending=False).head(k)
        g = gt[(gt["category"] == c) & (~gt["detected"])].sort_values("area", ascending=False).head(k)
        fig, axes = plt.subplots(2, k, figsize=(2.4 * k, 5.4), squeeze=False)
        for j in range(k):
            for r in range(2):
                axes[r, j].set_axis_off()
            if j < len(f):
                row = f.iloc[j]
                b = json.loads(row["bbox"]) if isinstance(row["bbox"], str) else row["bbox"]
                axes[0, j].imshow(crop(row["image_id"], [b[0], b[1], b[0] + b[2], b[1] + b[3]]))
                axes[0, j].set_title(f"FP {row['type'][:5]} {row['score']:.2f}", fontsize=8)
            if j < len(g):
                row = g.iloc[j]
                b = json.loads(row["bbox_xyxy"]) if isinstance(row["bbox_xyxy"], str) else row["bbox_xyxy"]
                axes[1, j].imshow(crop(row["image_id"], b))
                axes[1, j].set_title(f"missed {row['size']} IoU{row['best_iou_any']:.2f}", fontsize=8)
        fig.suptitle(f"{C.MODEL_LABELS[model_name]} - {c}: top false positives (row 1, magenta = prediction) "
                     f"and missed GT (row 2, magenta = GT)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{split}_{c}_gallery.png"), dpi=150)
        plt.close(fig)


# ==========================================================================
# speed
# ==========================================================================
def benchmark_speed(seed=0, split="test", n=C.SPEED_N_IMAGES, warmup=C.SPEED_WARMUP):
    """Batch 1, FP32, same GPU/session for all; end-to-end from decoded image in RAM to final
    detections (resize/normalise + forward + NMS). Image decoding from disk is excluded."""
    import torch
    ev = evaluator(split)
    files = [ev.images[i]["file_name"] for i in ev.img_ids[:n]]
    imgs_bgr = [cv2.imread(os.path.join(P.img_dir(split), f)) for f in files]
    rows = []
    for m in MODELS:
        if not os.path.exists(os.path.join(C.run_dir(m, seed, "checkpoints"), "best.pt")):
            continue
        model = load_model(m, seed)
        times = []
        if C.MODEL_CFG[m]["family"] == "ultralytics":
            def run(im):
                model.predict(im, imgsz=C.MODEL_CFG[m]["imgsz"], conf=C.INFER["conf_for_map"], iou=C.INFER["nms_iou"],
                              max_det=C.INFER["max_det"], half=False, verbose=False)
        else:
            @torch.no_grad()
            def run(im):
                t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255).cuda()
                model([t])
        for im in imgs_bgr[:warmup]:
            run(im)
        torch.cuda.synchronize()
        for im in imgs_bgr:
            t0 = time.perf_counter()
            run(im)
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - t0))
        params = (sum(p.numel() for p in model.model.parameters()) if C.MODEL_CFG[m]["family"] == "ultralytics"
                  else sum(p.numel() for p in model.parameters()))
        rows.append({"model": C.MODEL_LABELS[m], "input": C.MODEL_CFG[m]["imgsz"], "params_M": params / 1e6,
                     "ms_mean": np.mean(times), "ms_std": np.std(times), "ms_median": np.median(times),
                     "FPS": 1000 / np.mean(times), "n_images": len(times)})
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows).round(2)
    df["gpu"] = environment_snapshot().get("gpu_name")
    df.to_csv(os.path.join(C.COMPARISON_DIR, "speed_benchmark.csv"), index=False)
    return df


def efficiency_table(seed=0, n=50, warmup=10):
    """One consolidated efficiency row per model: Params (M), FP32 checkpoint size (MB),
    GFLOPs (thop, batch=1, native imgsz), Time (ms/img) and FPS (batch 1, FP32, same GPU,
    same protocol as benchmark_speed). This is the single source for the thesis efficiency
    table -- it replaces any ad-hoc GFLOPs/weights cell."""
    import torch
    try:
        from thop import profile as thop_profile
    except ImportError:
        os.system("pip install -q thop")
        from thop import profile as thop_profile

    rows = []
    for m in MODELS:
        ck = os.path.join(C.run_dir(m, seed, "checkpoints"), "best.pt")
        if not os.path.exists(ck):
            continue
        size_mb = os.path.getsize(ck) / (1024 ** 2)
        imgsz = C.MODEL_CFG[m]["imgsz"]
        family = C.MODEL_CFG[m]["family"]
        model = load_model(m, seed)
        net = model.model.eval() if family == "ultralytics" else model.eval()
        device = next(net.parameters()).device
        n_params = sum(p.numel() for p in net.parameters())
        if family == "ultralytics":
            dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
            flops, _ = thop_profile(net, inputs=(dummy,), verbose=False)
        else:
            dummy = [torch.randn(3, imgsz, imgsz, device=device)]
            flops, _ = thop_profile(net, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9

        imgs = [torch.randn(3, imgsz, imgsz) for _ in range(n + warmup)]

        def run_one(img):
            if family == "ultralytics":
                model.predict(img.permute(1, 2, 0).numpy(), imgsz=imgsz, conf=C.INFER["conf_for_map"],
                              iou=C.INFER["nms_iou"], max_det=C.INFER["max_det"], half=False, verbose=False)
            else:
                with torch.no_grad():
                    model([img.to(device)])

        for img in imgs[:warmup]:
            run_one(img)
        torch.cuda.synchronize()
        times = []
        for img in imgs[warmup:]:
            t0 = time.perf_counter()
            run_one(img)
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - t0))

        rows.append({"Model": C.MODEL_LABELS[m], "Params (M)": round(n_params / 1e6, 1),
                     "FP32 weights (MB)": round(size_mb, 1), "GFLOPs": round(gflops, 1),
                     "Time (ms/img)": round(float(np.mean(times)), 2), "FPS": round(1000 / float(np.mean(times)), 1)})
        del model, net
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.COMPARISON_DIR, "efficiency_table.csv"), index=False)
    return df


# ==========================================================================
# master summary table (everything the thesis results chapter needs, in one place)
# ==========================================================================
def master_table(split="test", efficiency=None):
    """Overall table: Precision/Recall/F1/mAP50/mAP50-95 (mean +- std over seeds) + Params (M),
    FP32 weights (MB), GFLOPs, Time (ms/img), FPS (if `efficiency` df is given, e.g. from
    efficiency_table()). Per-class table: Precision/Recall/F1/AP50/AP50-95 for every class.
    Saves {split}_master_summary_table.csv and {split}_master_per_class_table.csv."""
    overall, tabs = compare_models(split)
    if overall.empty:
        return overall, pd.DataFrame()
    overall = overall.reset_index().rename(columns={"index": "Model"})
    overall = overall.rename(columns={"macro_P": "Precision (%)", "macro_R": "Recall (%)",
                                       "macro_F1": "F1 (%)", "mAP50": "mAP50 (%)", "mAP50_95": "mAP50-95 (%)"})
    keep = ["Model", "Precision (%)", "Recall (%)", "F1 (%)", "mAP50 (%)", "mAP50-95 (%)", "seeds", "best_epochs"]
    overall = overall[[c for c in keep if c in overall.columns]]
    if efficiency is None:
        eff_path = os.path.join(C.COMPARISON_DIR, "efficiency_table.csv")
        efficiency = pd.read_csv(eff_path) if os.path.exists(eff_path) else None
    if efficiency is not None and not efficiency.empty:
        overall = overall.merge(efficiency, on="Model", how="left")

    rows = []
    for m_label in tabs["AP50"].columns:
        for c in C.CLASSES:
            if c not in tabs["AP50"].index:
                continue
            rows.append({"Model": m_label, "Class": c,
                         "Precision (%)": tabs["precision"].loc[c, m_label],
                         "Recall (%)": tabs["recall"].loc[c, m_label],
                         "F1 (%)": tabs["F1"].loc[c, m_label],
                         "AP50 (%)": tabs["AP50"].loc[c, m_label],
                         "AP50-95 (%)": tabs["AP50_95"].loc[c, m_label]})
    per_class = pd.DataFrame(rows)
    overall.to_csv(os.path.join(C.COMPARISON_DIR, f"{split}_master_summary_table.csv"), index=False)
    per_class.to_csv(os.path.join(C.COMPARISON_DIR, f"{split}_master_per_class_table.csv"), index=False)
    return overall, per_class


# ==========================================================================
# reproducibility appendix
# ==========================================================================
def reproducibility_appendix():
    L = ["# Reproducibility appendix\n"]
    si = load_json(os.path.join(C.SPLITS_DIR, "split_info.json"), {})
    hashes = load_json(os.path.join(C.SPLITS_DIR, "manifest_hashes.json"), {})
    L += ["## Data\n",
          "- Source: RDD2022 (Kaggle `aliabdelmenam/rdd-2022`), classes " + ", ".join(C.CLASSES) + "; Kaggle class 3 (faded road markings) excluded; "
          f"other labels dropped; excluded subsets: {', '.join(C.EXCLUDE_PREFIXES)}.",
          f"- Split: {si.get('stratification')}; ratios {si.get('ratios')}; split seed {si.get('split_seed')}. "
          "File lists: `results/00_dataset/*_manifest.csv`.",
          "- SHA-256 of split files: " + "; ".join(f"`{k}` {v[:16]}…" for k, v in hashes.items()),
          "- Albania held-out set: Roboflow annotation, never used for training, model selection or thresholds.\n"]
    cc = os.path.join(C.SPLITS_DIR, "class_counts.csv")
    if os.path.exists(cc):
        L += [pd.read_csv(cc).to_markdown(index=False), ""]
    L += ["## Shared protocol (identical for all models)\n",
          f"| Setting | Value |\n|---|---|\n"
          f"| Epochs | {C.EPOCHS} |\n| Effective batch | {C.EFFECTIVE_BATCH} |\n"
          f"| Optimizer | SGD, momentum {C.MOMENTUM}, Nesterov, weight decay {C.WEIGHT_DECAY} (not on bias/norm) |\n"
          f"| LR schedule | linear warmup {C.WARMUP_EPOCHS} epochs (≥{C.MIN_WARMUP_ITERS} it), cosine to {C.FINAL_LR_FRAC}×base, per epoch |\n"
          f"| Mixed precision / grad clip / EMA | AMP / {C.GRAD_CLIP_NORM} / decay {C.EMA_DECAY}, tau {C.EMA_TAU} |\n"
          f"| Augmentation | h-flip p={C.AUG['hflip_p']}, HSV value ±{C.AUG['brightness']}, scale ±{C.AUG['scale']} (fill 114); no mosaic/mixup/translate/rotation |\n"
          f"| Pretraining | COCO detection weights for all models |\n"
          f"| Model selection | best val {C.SELECTION_METRIC} (shared evaluator, EMA weights) |\n"
          f"| Inference (AP) | conf≥{C.INFER['conf_for_map']}, class-wise NMS IoU {C.INFER['nms_iou']}, max {C.INFER['max_det']} det |\n"
          f"| Operating point (P/R/F1) | conf threshold maximising macro-F1 on val, IoU {C.F1_IOU} |\n"
          f"| Seeds | {C.SEEDS}; bootstrap B={C.BOOTSTRAP_B} (seed {C.BOOTSTRAP_SEED}) |\n"
          f"| Speed | batch 1, FP32, {C.SPEED_WARMUP} warm-up, {C.SPEED_N_IMAGES} images, incl. pre/post-processing |\n",
          "## Per-model settings\n",
          "| Model | Input | Pretrained weights | Base LR | Batch × accum | Params (M) | Best epochs | Val thresholds |\n|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        infos = [load_json(os.path.join(C.run_dir(m, s, "logs"), "run_info.json")) for s in C.SEEDS]
        infos = [i for i in infos if i]
        sel = [load_json(os.path.join(C.run_dir(m, s, "logs"), "selection.json"), {}).get("best_epoch") for s in C.SEEDS]
        thr = [load_json(os.path.join(C.run_dir(m, s, "metrics"), "operating_threshold.json"), {}).get("threshold") for s in C.SEEDS]
        cfg = C.MODEL_CFG[m]
        npar = infos[0].get("n_params", 0) / 1e6 if infos else float("nan")
        acc = infos[0].get("grad_accum", "?") if infos else "?"
        L.append(f"| {C.MODEL_LABELS[m]} | {cfg['imgsz']} | {cfg['weights']} | {cfg['base_lr']} | "
                 f"{cfg['batch']} × {acc} | {npar:.1f} | {sel} | {thr} |")
    L += ["\n## Hardware and software (per run)\n",
          "| Model | Seed | GPU | Driver / CUDA | torch | torchvision | ultralytics | pycocotools | Python |\n|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for s in C.SEEDS:
            i = load_json(os.path.join(C.run_dir(m, s, "logs"), "run_info.json"))
            if not i:
                continue
            e = i["environment"]
            L.append(f"| {C.MODEL_LABELS[m]} | {s} | {e.get('gpu_name')} | {e.get('nvidia_smi', '').split(',')[1:2]} / "
                     f"{e.get('cuda_version_torch')} | {e.get('ver_torch')} | {e.get('ver_torchvision')} | "
                     f"{e.get('ver_ultralytics')} | {e.get('ver_pycocotools')} | {e.get('python')} |")
    L += ["\n## Known, documented differences\n",
          "- SSD300 runs at its native 300×300 input (anchor design); YOLOv8m and Faster R-CNN use 640 px (longest side).",
          "- Base learning rates are each architecture's standard value for batch 16; schedule shape is shared.",
          "- GPU training is not bit-exact even with fixed seeds (non-deterministic CUDA kernels such as roi_align backward); "
          "only one training seed per model was run, so this run-to-run variation is not captured.",
          "- Faster R-CNN uses FrozenBatchNorm in the backbone (torchvision default when fine-tuning)."]
    p = os.path.join(C.COMPARISON_DIR, "reproducibility_appendix.md")
    with open(p, "w") as f:
        f.write("\n".join(L))
    return p
