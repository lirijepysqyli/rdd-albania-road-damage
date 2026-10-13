"""
fair_eval.py - ONE evaluation implementation used for all three models.

Input detections, for every model, are a list of dicts in original-image pixel coordinates:
    {"image_id": int, "category_id": 1..5, "bbox": [x, y, w, h], "score": float}

Provides
  * COCO metrics via pycocotools (mAP50, mAP50-95, AP_S/M/L, per-class AP / AP50 / AP by size)
  * an exact re-implementation of COCO AP on pre-computed match tables, used for
      - bootstrap 95% CIs (resampling test images)
      - per-country AP
      - AP vs IoU-threshold sweep (0.1 ... 0.95)  -> D20 localisation analysis
  * P / R / F1 per class at an operating threshold chosen on VAL (max macro-F1)
  * class-agnostic confusion matrix incl. background
  * TIDE-style error counts per class (cls / loc / both / duplicate / background / missed)
  * recall by object size per class
"""
import collections
import contextlib
import io
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

from . import config as C

IOU_THRS_COCO = np.round(np.linspace(0.5, 0.95, 10), 2)
IOU_THRS_SWEEP = np.round(np.array([0.1, 0.2, 0.3, 0.4] + list(IOU_THRS_COCO)), 2)
REC_THRS = np.linspace(0.0, 1.0, 101)
SIZE_BINS = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}
ERROR_TYPES = ["classification", "localization", "both", "duplicate", "background"]


def _xywh2xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], 1)


def box_iou(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


class Evaluator:
    def __init__(self, gt_json):
        self.gt_json = gt_json
        self.gt = json.load(open(gt_json)) if isinstance(gt_json, str) else gt_json
        self.cat_ids = [c["id"] for c in sorted(self.gt["categories"], key=lambda c: c["id"])]
        self.cat_names = {c["id"]: c["name"] for c in self.gt["categories"]}
        self.img_ids = [im["id"] for im in self.gt["images"]]
        self.img_index = {iid: i for i, iid in enumerate(self.img_ids)}
        self.images = {im["id"]: im for im in self.gt["images"]}
        self.gt_boxes = collections.defaultdict(lambda: np.zeros((0, 4)))  # (img, cat) -> xyxy
        self.gt_area = collections.defaultdict(lambda: np.zeros(0))
        tmp_b, tmp_a = collections.defaultdict(list), collections.defaultdict(list)
        for a in self.gt["annotations"]:
            tmp_b[(a["image_id"], a["category_id"])].append(a["bbox"])
            tmp_a[(a["image_id"], a["category_id"])].append(a.get("area", a["bbox"][2] * a["bbox"][3]))
        for k in tmp_b:
            self.gt_boxes[k] = _xywh2xyxy(tmp_b[k])
            self.gt_area[k] = np.asarray(tmp_a[k], dtype=np.float64)
        # n_gt[img_idx, cat_idx]
        self.n_gt = np.zeros((len(self.img_ids), len(self.cat_ids)), dtype=np.float64)
        for (iid, cid), b in self.gt_boxes.items():
            self.n_gt[self.img_index[iid], self.cat_ids.index(cid)] = len(b)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _group_dets(dets):
        g = collections.defaultdict(list)
        for d in dets:
            g[(d["image_id"], d["category_id"])].append(d)
        return g

    # ------------------------------------------------------------------ pycocotools
    def coco_metrics(self, dets):
        """Official COCO numbers. Returns dict with overall and per-class values (in %)."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        out = {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "AP_small": 0.0, "AP_medium": 0.0,
               "AP_large": 0.0, "per_class": {}}
        if len(dets) == 0:
            return out
        with contextlib.redirect_stdout(io.StringIO()):
            cg = COCO()
            cg.dataset = json.loads(json.dumps(self.gt))
            cg.createIndex()
            cd = cg.loadRes([dict(d) for d in dets])
            E = COCOeval(cg, cd, "bbox")
            E.params.maxDets = [1, 10, C.INFER["max_det"]]
            E.evaluate()
            E.accumulate()
            E.summarize()
        s = E.stats
        out.update(mAP50_95=100 * s[0], mAP50=100 * s[1], mAP75=100 * s[2],
                   AP_small=100 * s[3], AP_medium=100 * s[4], AP_large=100 * s[5])
        P = E.eval["precision"]  # [T, R, K, A, M]
        for k, cid in enumerate(E.params.catIds):
            def ap(p):
                p = p[p > -1]
                return float(100 * p.mean()) if p.size else float("nan")
            out["per_class"][self.cat_names[cid]] = {
                "AP50_95": ap(P[:, :, k, 0, -1]),
                "AP50": ap(P[0, :, k, 0, -1]),
                "AP75": ap(P[5, :, k, 0, -1]),
                "AP_small": ap(P[:, :, k, 1, -1]),
                "AP_medium": ap(P[:, :, k, 2, -1]),
                "AP_large": ap(P[:, :, k, 3, -1]),
            }
        return out

    # ------------------------------------------------------------------ match tables
    def match_tables(self, dets, iou_thrs=IOU_THRS_SWEEP):
        """COCO-style greedy matching (per image, per class, by descending score, each det
        takes the highest-IoU unmatched GT above the threshold) for several IoU thresholds.
        Returns per-class dict: scores (nd,), img (nd,), tp (T, nd) bool."""
        g = self._group_dets(dets)
        maxd = C.INFER["max_det"]
        tables = {}
        for cid in self.cat_ids:
            sc, im, tp = [], [], []
            for iid in self.img_ids:
                ds = g.get((iid, cid))
                if not ds:
                    continue
                ds = sorted(ds, key=lambda d: -d["score"])[:maxd]
                db = _xywh2xyxy([d["bbox"] for d in ds])
                gb = self.gt_boxes[(iid, cid)]
                ious = box_iou(db, gb)
                t = np.zeros((len(iou_thrs), len(ds)), dtype=bool)
                for ti, thr in enumerate(iou_thrs):
                    used = np.zeros(len(gb), dtype=bool)
                    for di in range(len(ds)):
                        if len(gb) == 0:
                            break
                        cand = np.where(~used & (ious[di] >= thr - 1e-10), ious[di], -1)
                        j = int(np.argmax(cand))
                        if cand[j] >= 0:
                            used[j] = True
                            t[ti, di] = True
                sc.extend(d["score"] for d in ds)
                im.extend([self.img_index[iid]] * len(ds))
                tp.append(t)
            sc = np.asarray(sc, dtype=np.float64)
            order = np.argsort(-sc, kind="mergesort")
            tables[cid] = {
                "scores": sc[order],
                "img": np.asarray(im, dtype=np.int64)[order],
                "tp": (np.concatenate(tp, 1)[:, order] if tp else np.zeros((len(iou_thrs), 0), bool)),
                "iou_thrs": np.asarray(iou_thrs),
            }
        return tables

    def ap_from_tables(self, tables, img_weights=None, iou_subset=None):
        """COCO 101-point interpolated AP per class, per IoU threshold.
        img_weights: (n_images,) multiplicity of each image (bootstrap) or 0/1 mask (subset).
        Returns {cid: array over IoU thresholds} ; NaN when the class has no GT in the sample."""
        w_img = np.ones(len(self.img_ids)) if img_weights is None else np.asarray(img_weights, float)
        res = {}
        for k, cid in enumerate(self.cat_ids):
            tb = tables[cid]
            thr_idx = (np.arange(len(tb["iou_thrs"])) if iou_subset is None
                       else np.array([int(np.argmin(np.abs(tb["iou_thrs"] - t))) for t in iou_subset]))
            npig = float((self.n_gt[:, k] * w_img).sum())
            if npig == 0:
                res[cid] = np.full(len(thr_idx), np.nan)
                continue
            if tb["scores"].size == 0:
                res[cid] = np.zeros(len(thr_idx))
                continue
            w = w_img[tb["img"]]
            keep = w > 0
            w = w[keep]
            aps = np.zeros(len(thr_idx))
            for o, ti in enumerate(thr_idx):
                tpw = tb["tp"][ti][keep] * w
                fpw = (~tb["tp"][ti][keep]) * w
                tpc, fpc = np.cumsum(tpw), np.cumsum(fpw)
                rc = tpc / npig
                pr = tpc / np.maximum(tpc + fpc, np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                inds = np.searchsorted(rc, REC_THRS, side="left")
                q = np.zeros(len(REC_THRS))
                valid = inds < len(pr)
                q[valid] = pr[inds[valid]]
                aps[o] = q.mean()
            res[cid] = 100 * aps
        return res

    @staticmethod
    def summarise_ap(ap_by_class, iou_thrs):
        """mAP50 and mAP50-95 from ap_from_tables output."""
        thr = np.asarray(iou_thrs)
        i50 = int(np.argmin(np.abs(thr - 0.5)))
        coco_idx = [int(np.argmin(np.abs(thr - t))) for t in IOU_THRS_COCO]
        arr = np.array([v for v in ap_by_class.values()])  # K x T
        return {"mAP50": float(np.nanmean(arr[:, i50])),
                "mAP50_95": float(np.nanmean(arr[:, coco_idx]))}

    # ------------------------------------------------------------------ P / R / F1
    def pr_f1(self, tables, thr, iou=C.F1_IOU, img_weights=None):
        w_img = np.ones(len(self.img_ids)) if img_weights is None else np.asarray(img_weights, float)
        out = {}
        for k, cid in enumerate(self.cat_ids):
            tb = tables[cid]
            ti = int(np.argmin(np.abs(tb["iou_thrs"] - iou)))
            sel = tb["scores"] >= thr
            w = w_img[tb["img"][sel]]
            tp = float((tb["tp"][ti][sel] * w).sum())
            fp = float(((~tb["tp"][ti][sel]) * w).sum())
            ngt = float((self.n_gt[:, k] * w_img).sum())
            fn = ngt - tp
            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / ngt if ngt > 0 else float("nan")
            f1 = 2 * p * r / (p + r) if (ngt > 0 and p + r > 0) else (0.0 if ngt > 0 else float("nan"))
            out[self.cat_names[cid]] = {"TP": tp, "FP": fp, "FN": fn, "n_gt": ngt,
                                        "precision": 100 * p, "recall": 100 * r if r == r else r,
                                        "F1": 100 * f1 if f1 == f1 else f1}
        vals = [v for v in out.values() if v["n_gt"] > 0]
        out["macro"] = {k: float(np.mean([v[k] for v in vals])) for k in ("precision", "recall", "F1")}
        return out

    def choose_threshold(self, tables, grid=np.round(np.arange(0.01, 1.0, 0.01), 2)):
        """Operating point = confidence threshold maximising macro-F1 (use on VAL only)."""
        best_t, best_f = 0.5, -1
        curve = []
        for t in grid:
            f = self.pr_f1(tables, t)["macro"]["F1"]
            curve.append((float(t), f))
            if f > best_f:
                best_t, best_f = float(t), f
        return best_t, best_f, curve

    def pr_curves(self, tables, iou=0.5):
        curves = {}
        for k, cid in enumerate(self.cat_ids):
            tb = tables[cid]
            ti = int(np.argmin(np.abs(tb["iou_thrs"] - iou)))
            npig = self.n_gt[:, k].sum()
            if npig == 0 or tb["scores"].size == 0:
                continue
            tpc = np.cumsum(tb["tp"][ti])
            fpc = np.cumsum(~tb["tp"][ti])
            curves[self.cat_names[cid]] = (tpc / npig, tpc / np.maximum(tpc + fpc, 1), tb["scores"])
        return curves

    # ------------------------------------------------------------------ bootstrap
    def bootstrap(self, tables, thr, B=C.BOOTSTRAP_B, seed=C.BOOTSTRAP_SEED, alpha=0.05):
        """Image-level non-parametric bootstrap. Returns point estimates + percentile CIs for
        mAP50, mAP50-95, per-class AP50 / AP50-95 / F1, macro-F1."""
        rng = np.random.default_rng(seed)
        n = len(self.img_ids)
        coco_only = IOU_THRS_COCO
        samples = collections.defaultdict(list)
        for _ in range(B):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            ap = self.ap_from_tables(tables, w, iou_subset=coco_only)
            arr = np.array(list(ap.values()))
            samples["mAP50"].append(np.nanmean(arr[:, 0]))
            samples["mAP50_95"].append(np.nanmean(arr))
            f = self.pr_f1(tables, thr, img_weights=w)
            samples["macro_F1"].append(f["macro"]["F1"])
            for k, cid in enumerate(self.cat_ids):
                nm = self.cat_names[cid]
                samples[f"{nm}_AP50"].append(arr[k, 0])
                samples[f"{nm}_AP50_95"].append(np.nanmean(arr[k]))
                samples[f"{nm}_F1"].append(f[nm]["F1"])
        ap = self.ap_from_tables(tables, iou_subset=coco_only)
        arr = np.array(list(ap.values()))
        f = self.pr_f1(tables, thr)
        point = {"mAP50": np.nanmean(arr[:, 0]), "mAP50_95": np.nanmean(arr), "macro_F1": f["macro"]["F1"]}
        for k, cid in enumerate(self.cat_ids):
            nm = self.cat_names[cid]
            point[f"{nm}_AP50"] = arr[k, 0]
            point[f"{nm}_AP50_95"] = np.nanmean(arr[k])
            point[f"{nm}_F1"] = f[nm]["F1"]
        out = {}
        for key, vals in samples.items():
            v = np.asarray(vals, float)
            v = v[~np.isnan(v)]
            lo, hi = (np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if v.size else (np.nan, np.nan))
            out[key] = {"point": float(point[key]), "ci_low": float(lo), "ci_high": float(hi)}
        return out

    # ------------------------------------------------------------------ per-country
    def per_country(self, tables):
        countries = collections.defaultdict(list)
        for iid in self.img_ids:
            countries[self.images[iid].get("country", "unknown")].append(self.img_index[iid])
        rows = {}
        for c, idx in sorted(countries.items()):
            mask = np.zeros(len(self.img_ids))
            mask[idx] = 1
            ap = self.ap_from_tables(tables, mask, iou_subset=IOU_THRS_COCO)
            row = {"images": len(idx), **self.summarise_ap(ap, IOU_THRS_COCO)}
            for cid in self.cat_ids:
                row[f"{self.cat_names[cid]}_AP50"] = float(ap[cid][0])
            rows[c] = row
        return rows

    # ------------------------------------------------------------------ error analysis
    def analyse(self, dets, thr, iou=C.F1_IOU):
        """Detailed analysis at the operating threshold:
        confusion matrix, TIDE-style error counts, per-GT and per-FP records."""
        K = len(self.cat_ids)
        cidx = {cid: i for i, cid in enumerate(self.cat_ids)}
        cm = np.zeros((K + 1, K + 1), dtype=np.int64)       # rows = true, cols = predicted, K = background
        err = {self.cat_names[c]: collections.Counter() for c in self.cat_ids}
        confused_with = {self.cat_names[c]: collections.Counter() for c in self.cat_ids}
        gt_records, fp_records = [], []
        dets_by_img = collections.defaultdict(list)
        for d in dets:
            if d["score"] >= thr:
                dets_by_img[d["image_id"]].append(d)
        for iid in self.img_ids:
            ds = sorted(dets_by_img.get(iid, []), key=lambda d: -d["score"])[:C.INFER["max_det"]]
            gtb, gtc, gta = [], [], []
            for cid in self.cat_ids:
                b = self.gt_boxes[(iid, cid)]
                gtb.append(b)
                gtc += [cid] * len(b)
                gta += list(self.gt_area[(iid, cid)])
            gtb = np.concatenate(gtb) if gtb else np.zeros((0, 4))
            gtc, gta = np.asarray(gtc), np.asarray(gta)
            db = _xywh2xyxy([d["bbox"] for d in ds]) if ds else np.zeros((0, 4))
            dc = np.asarray([d["category_id"] for d in ds])
            ious = box_iou(db, gtb)

            # (1) class-aware matching -> TP/FP per class (same rule as match_tables)
            matched_gt = np.zeros(len(gtb), bool)
            det_tp = np.zeros(len(ds), bool)
            for di in range(len(ds)):
                same = (gtc == dc[di]) & ~matched_gt
                cand = np.where(same & (ious[di] >= iou - 1e-10), ious[di], -1) if len(gtb) else np.array([-1])
                j = int(np.argmax(cand))
                if cand[j] >= 0:
                    matched_gt[j] = True
                    det_tp[di] = True
            # (2) TIDE-style categorisation of every false positive
            for di in np.where(~det_tp)[0]:
                c = self.cat_names[dc[di]]
                same_mask = gtc == dc[di]
                iou_same = ious[di][same_mask].max() if same_mask.any() else 0.0
                other_mask = ~same_mask
                iou_other = ious[di][other_mask].max() if other_mask.any() else 0.0
                if iou_same >= iou:
                    t = "duplicate"
                elif iou_other >= iou:
                    t = "classification"
                    j = np.where(other_mask)[0][np.argmax(ious[di][other_mask])]
                    confused_with[c][self.cat_names[gtc[j]]] += 1
                elif iou_same >= 0.1:
                    t = "localization"
                elif iou_other >= 0.1:
                    t = "both"
                else:
                    t = "background"
                err[c][t] += 1
                fp_records.append({"image_id": iid, "category": c, "score": ds[di]["score"],
                                   "bbox": ds[di]["bbox"], "type": t,
                                   "iou_same": float(iou_same), "iou_other": float(iou_other)})
            # (3) missed GT
            for j in range(len(gtb)):
                c = self.cat_names[gtc[j]]
                best_any = ious[:, j].max() if len(ds) else 0.0
                if not matched_gt[j]:
                    err[c]["missed" if best_any < 0.1 else "missed_but_overlapped"] += 1
                size = next(s for s, (lo, hi) in SIZE_BINS.items() if lo <= gta[j] < hi)
                gt_records.append({"image_id": iid, "category": c, "area": float(gta[j]), "size": size,
                                   "country": self.images[iid].get("country", "unknown"),
                                   "detected": bool(matched_gt[j]), "best_iou_any": float(best_any),
                                   "bbox_xyxy": gtb[j].tolist()})
            # (4) class-agnostic confusion matrix (greedy by score, IoU >= iou)
            used = np.zeros(len(gtb), bool)
            for di in range(len(ds)):
                cand = np.where(~used & (ious[di] >= iou - 1e-10), ious[di], -1) if len(gtb) else np.array([-1])
                j = int(np.argmax(cand))
                if cand[j] >= 0:
                    used[j] = True
                    cm[cidx[gtc[j]], cidx[dc[di]]] += 1
                else:
                    cm[K, cidx[dc[di]]] += 1
            for j in np.where(~used)[0]:
                cm[cidx[gtc[j]], K] += 1

        # recall by size
        size_recall = {}
        for c in [self.cat_names[x] for x in self.cat_ids]:
            size_recall[c] = {}
            for s in SIZE_BINS:
                rs = [g for g in gt_records if g["category"] == c and g["size"] == s]
                size_recall[c][s] = {"n_gt": len(rs),
                                     "recall": 100 * np.mean([g["detected"] for g in rs]) if rs else float("nan")}
        names = [self.cat_names[x] for x in self.cat_ids] + ["background"]
        return {"threshold": thr, "confusion_matrix": cm.tolist(), "cm_labels": names,
                "errors": {c: dict(v) for c, v in err.items()},
                "classification_confused_with": {c: dict(v) for c, v in confused_with.items()},
                "size_recall": size_recall, "gt_records": gt_records, "fp_records": fp_records}


# ----------------------------------------------------------------------------
# convenience: full evaluation of one prediction file -> dict (saved by caller)
# ----------------------------------------------------------------------------
def full_evaluation(evaluator, dets, thr, do_bootstrap=True):
    tables = evaluator.match_tables(dets)
    coco = evaluator.coco_metrics(dets)
    own = evaluator.summarise_ap(evaluator.ap_from_tables(tables), IOU_THRS_SWEEP)
    # sanity check: our AP implementation must agree with pycocotools
    agree = abs(own["mAP50_95"] - coco["mAP50_95"]) < 0.5 and abs(own["mAP50"] - coco["mAP50"]) < 0.5
    sweep = {evaluator.cat_names[cid]: dict(zip([float(t) for t in IOU_THRS_SWEEP], map(float, v)))
             for cid, v in evaluator.ap_from_tables(tables).items()}
    res = {
        "coco": coco,
        "own_ap_check": {**own, "agrees_with_pycocotools": bool(agree)},
        "threshold": thr,
        "prf1": evaluator.pr_f1(tables, thr),
        "ap_vs_iou": sweep,
        "per_country": evaluator.per_country(tables),
        "analysis": evaluator.analyse(dets, thr),
    }
    if do_bootstrap:
        res["bootstrap"] = evaluator.bootstrap(tables, thr)
    return res, tables
