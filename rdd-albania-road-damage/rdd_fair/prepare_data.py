"""
Builds ONE canonical dataset that all three models read.

First ever session:
  unzip -> parse annotations (Pascal-VOC XML or YOLO txt, auto-detected) -> keep 5 classes
  -> drop China_Drone -> stratified 70/15/15 split (seed 42) -> save COCO jsons + manifests
  + SHA-256 hashes to results/00_dataset/ on Drive.

Every later session:
  unzip -> rebuild the local dataset FROM THE SAVED COCO JSONS (no re-splitting), verify hashes.

YOLO .txt labels are always generated from the COCO jsons, and box counts are asserted equal,
so SSD, Faster R-CNN and YOLO see exactly the same boxes.
"""
import collections
import csv
import glob
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

from . import config as C
from .utils import load_json, save_json, sha256_file

SPLITS = ("train", "val", "test")
KNOWN_COUNTRIES = ["China_MotorBike", "China_Drone", "Czech", "India", "Japan",
                   "Norway", "United_States"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def canon_class(name):
    return C.CLASS_ALIASES.get(str(name).strip().lower())


def country_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    for c in sorted(KNOWN_COUNTRIES, key=len, reverse=True):
        if stem.startswith(c + "_") or stem == c:
            return c
    m = re.match(r"^(.*?)_\d+$", stem)
    if m:
        return m.group(1)
    for c in sorted(KNOWN_COUNTRIES, key=len, reverse=True):
        if f"/{c}/" in path.replace("\\", "/"):
            return c
    return "unknown"


def unzip_once(zip_path=C.RDD_ZIP, dest=C.LOCAL_ROOT):
    marker = os.path.join(dest, ".unzipped")
    if os.path.exists(marker):
        return dest
    os.makedirs(dest, exist_ok=True)
    print(f"Unzipping {zip_path} -> {dest} (local disk, fast I/O) ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    open(marker, "w").close()
    return dest


def _find_yolo_names(root):
    """Return list of class names from a data.yaml / classes.txt found under root."""
    for y in glob.glob(os.path.join(root, "**", "*.yaml"), recursive=True):
        try:
            import yaml
            d = yaml.safe_load(open(y))
            names = d.get("names") if isinstance(d, dict) else None
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            if names:
                print(f"Class names from {y}: {names}")
                return list(names)
        except Exception:
            continue
    for t in glob.glob(os.path.join(root, "**", "classes.txt"), recursive=True):
        names = [l.strip() for l in open(t) if l.strip()]
        print(f"Class names from {t}: {names}")
        return names
    print("No class-name file in the zip -> using config.YOLO_CLASS_NAMES:", C.YOLO_CLASS_NAMES)
    return list(C.YOLO_CLASS_NAMES)


def _img_size(path):
    with Image.open(path) as im:
        return im.size  # (w, h)


def _clip_box(x1, y1, x2, y2, w, h):
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


# --------------------------------------------------------------------------
# scanning raw data
# --------------------------------------------------------------------------
def scan_raw(root):
    """Returns list of records {src, file_name, country, width, height, boxes:[(cls, x1,y1,x2,y2)]}
    and a stats dict."""
    images = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
              if p.lower().endswith(C.IMG_EXTS)]
    xmls = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True)}
    txts = [p for p in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True)
            if os.path.basename(p) != "classes.txt"]
    fmt = "voc" if xmls else ("yolo" if txts else None)
    if fmt is None:
        raise RuntimeError("No XML or YOLO txt annotations found in the zip.")
    print(f"Found {len(images)} images; annotation format = {fmt.upper()}")
    yolo_names = _find_yolo_names(root) if fmt == "yolo" else None

    stats = collections.Counter()
    dropped_names = collections.Counter()
    records, seen = [], {}

    # folders with no label files at all are unlabeled (e.g. official RDD2022 test) -> skip
    labeled_dirs = set()
    if fmt == "yolo":
        for t in txts:
            labeled_dirs.add(os.path.dirname(t).replace(os.sep + "labels", os.sep + "images"))

    for img in sorted(images):
        fn = os.path.basename(img)
        stem = os.path.splitext(fn)[0]
        if any(fn.startswith(p) for p in C.EXCLUDE_PREFIXES):
            stats["excluded_prefix"] += 1
            continue
        boxes = []
        if fmt == "voc":
            xp = xmls.get(stem)
            if xp is None:
                stats["no_annotation_skipped"] += 1
                continue
            r = ET.parse(xp).getroot()
            sz = r.find("size")
            try:
                w, h = int(float(sz.find("width").text)), int(float(sz.find("height").text))
                if w <= 0 or h <= 0:
                    raise ValueError
            except Exception:
                w, h = _img_size(img)
            for o in r.findall("object"):
                name = o.find("name").text
                cls = canon_class(name)
                if cls is None:
                    dropped_names[name] += 1
                    continue
                b = o.find("bndbox")
                bb = _clip_box(float(b.find("xmin").text), float(b.find("ymin").text),
                               float(b.find("xmax").text), float(b.find("ymax").text), w, h)
                if bb is None:
                    stats["degenerate_box"] += 1
                    continue
                boxes.append((cls, *bb))
        else:
            if os.path.dirname(img) not in labeled_dirs:
                stats["unlabeled_folder_skipped"] += 1
                continue
            lp = os.path.splitext(img.replace(os.sep + "images" + os.sep,
                                              os.sep + "labels" + os.sep))[0] + ".txt"
            w, h = _img_size(img)
            if os.path.exists(lp):
                for line in open(lp):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    name = yolo_names[int(float(parts[0]))]
                    cls = canon_class(name)
                    if cls is None:
                        dropped_names[name] += 1
                        continue
                    cx, cy, bw, bh = map(float, parts[1:5])
                    bb = _clip_box((cx - bw / 2) * w, (cy - bh / 2) * h,
                                   (cx + bw / 2) * w, (cy + bh / 2) * h, w, h)
                    if bb is None:
                        stats["degenerate_box"] += 1
                        continue
                    boxes.append((cls, *bb))
            else:
                stats["background_image_no_txt"] += 1
        if fn in seen:
            stats["duplicate_filename_skipped"] += 1
            continue
        seen[fn] = True
        records.append(dict(src=os.path.relpath(img, root), file_name=fn, country=country_of(img),
                            width=w, height=h, boxes=boxes))
    stats["dropped_other_classes"] = sum(dropped_names.values())
    print("Scan stats:", dict(stats))
    print("Dropped (non-target) class labels:", dict(dropped_names))
    return records, dict(stats), dict(dropped_names)


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------
def stratified_split(records, seed=C.SPLIT_SEED, ratios=C.SPLIT_RATIOS):
    """Stratify by (country, rarest target class present in image, or 'bg')."""
    freq = collections.Counter(b[0] for r in records for b in r["boxes"])
    def key(r):
        present = {b[0] for b in r["boxes"]}
        rare = min(present, key=lambda c: freq[c]) if present else "bg"
        return f"{r['country']}|{rare}"
    groups = collections.defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    rng = np.random.default_rng(seed)
    out = {s: [] for s in SPLITS}
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda r: r["file_name"])
        idx = rng.permutation(len(g))
        n = len(g)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        if n >= 3:  # make sure tiny groups still reach val and test
            n_va = max(n_va, 1)
            n_tr = min(n_tr, n - n_va - 1)
        for j, i in enumerate(idx):
            s = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
            out[s].append(g[i])
    return out


# --------------------------------------------------------------------------
# COCO / YOLO writers
# --------------------------------------------------------------------------
def records_to_coco(records):
    cats = [{"id": i + 1, "name": c, "supercategory": "road_damage"} for i, c in enumerate(C.CLASSES)]
    cid = {c: i + 1 for i, c in enumerate(C.CLASSES)}
    images, anns, aid = [], [], 1
    for iid, r in enumerate(sorted(records, key=lambda r: r["file_name"]), start=1):
        images.append({"id": iid, "file_name": r["file_name"], "width": r["width"],
                       "height": r["height"], "country": r["country"], "src": r["src"]})
        for cls, x1, y1, x2, y2 in r["boxes"]:
            w, h = x2 - x1, y2 - y1
            anns.append({"id": aid, "image_id": iid, "category_id": cid[cls],
                         "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2), "iscrowd": 0})
            aid += 1
    return {"images": images, "annotations": anns, "categories": cats}


def write_yolo_labels(coco, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    by_img = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    n = 0
    for im in coco["images"]:
        W, H = im["width"], im["height"]
        lines = []
        for a in by_img[im["id"]]:
            x, y, w, h = a["bbox"]
            lines.append(f"{a['category_id'] - 1} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                         f"{w / W:.6f} {h / H:.6f}")
            n += 1
        with open(os.path.join(label_dir, os.path.splitext(im["file_name"])[0] + ".txt"), "w") as f:
            f.write("\n".join(lines))
    return n


def _link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialise_split(coco, split, src_root):
    img_dir = os.path.join(C.DATA_ROOT, "images", split)
    os.makedirs(img_dir, exist_ok=True)
    for im in coco["images"]:
        _link(os.path.join(src_root, im["src"]), os.path.join(img_dir, im["file_name"]))
    n_yolo = write_yolo_labels(coco, os.path.join(C.DATA_ROOT, "labels", split))
    assert n_yolo == len(coco["annotations"]), f"{split}: YOLO/COCO box count mismatch"
    ann_dir = os.path.join(C.DATA_ROOT, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, f"{split}.json"), "w") as f:
        json.dump(coco, f)


def write_yolo_yaml():
    p = os.path.join(C.DATA_ROOT, "data.yaml")
    with open(p, "w") as f:
        f.write(f"path: {C.DATA_ROOT}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(C.CLASSES)))
    return p


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------
def count_table(cocos):
    rows = []
    for split, coco in cocos.items():
        cnt = collections.Counter(a["category_id"] for a in coco["annotations"])
        row = {"split": split, "images": len(coco["images"]),
               "background_images": len(coco["images"]) - len({a["image_id"] for a in coco["annotations"]})}
        for i, c in enumerate(C.CLASSES):
            row[c] = cnt.get(i + 1, 0)
        row["boxes_total"] = len(coco["annotations"])
        rows.append(row)
    return rows


def country_table(cocos):
    rows = []
    for split, coco in cocos.items():
        iid2c = {im["id"]: im.get("country", "unknown") for im in coco["images"]}
        img_c = collections.Counter(iid2c.values())
        box_c = collections.defaultdict(collections.Counter)
        for a in coco["annotations"]:
            box_c[iid2c[a["image_id"]]][a["category_id"]] += 1
        for country in sorted(img_c):
            row = {"split": split, "country": country, "images": img_c[country]}
            for i, c in enumerate(C.CLASSES):
                row[c] = box_c[country].get(i + 1, 0)
            rows.append(row)
    return rows


def _write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# main entry points
# --------------------------------------------------------------------------
def prepare_rdd(force_resplit=False):
    """Call at the top of every notebook. Returns path to YOLO data.yaml."""
    os.makedirs(C.SPLITS_DIR, exist_ok=True)
    src_root = unzip_once()
    hash_path = os.path.join(C.SPLITS_DIR, "manifest_hashes.json")
    saved = {s: os.path.join(C.SPLITS_DIR, f"rdd_{s}_coco.json") for s in SPLITS}

    if force_resplit or not all(os.path.exists(p) for p in saved.values()):
        print("No saved split found -> creating the fixed split (this happens only once).")
        records, stats, dropped = scan_raw(src_root)
        parts = stratified_split(records)
        cocos = {s: records_to_coco(parts[s]) for s in SPLITS}
        for s in SPLITS:
            with open(saved[s], "w") as f:
                json.dump(cocos[s], f)
            with open(os.path.join(C.SPLITS_DIR, f"{s}_manifest.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file_name", "source_path", "country", "width", "height", "n_boxes"])
                n_by = collections.Counter(a["image_id"] for a in cocos[s]["annotations"])
                for im in cocos[s]["images"]:
                    w.writerow([im["file_name"], im["src"], im["country"], im["width"],
                                im["height"], n_by.get(im["id"], 0)])
        hashes = {os.path.basename(p): sha256_file(p) for p in
                  list(saved.values()) + [os.path.join(C.SPLITS_DIR, f"{s}_manifest.csv") for s in SPLITS]}
        save_json(hashes, hash_path)
        save_json({"scan_stats": stats, "dropped_non_target_labels": dropped,
                   "split_seed": C.SPLIT_SEED, "ratios": C.SPLIT_RATIOS,
                   "stratification": "country x rarest target class present (or background)",
                   "excluded_prefixes": C.EXCLUDE_PREFIXES}, os.path.join(C.SPLITS_DIR, "split_info.json"))
        _write_csv(count_table(cocos), os.path.join(C.SPLITS_DIR, "class_counts.csv"))
        _write_csv(country_table(cocos), os.path.join(C.SPLITS_DIR, "country_counts.csv"))
    else:
        hashes = load_json(hash_path)
        for s in SPLITS:
            hv = sha256_file(saved[s])
            assert hv == hashes[os.path.basename(saved[s])], f"Split file {saved[s]} changed! Hash mismatch."
        cocos = {s: json.load(open(saved[s])) for s in SPLITS}
        print("Loaded saved split (hashes verified).")

    for s in SPLITS:
        materialise_split(cocos[s], s, src_root)
    yaml_path = write_yolo_yaml()
    for row in count_table(cocos):
        print(row)
    return yaml_path


def prepare_albania(src=C.ALBANIA_SRC):
    """Converts the Roboflow export (COCO json or YOLO folder) into the canonical format
    as split 'albania'. Never used for training or model selection."""
    out_json = os.path.join(C.SPLITS_DIR, "albania_coco.json")
    jsons = glob.glob(os.path.join(src, "**", "*.json"), recursive=True)
    records = []
    if jsons:
        jp = jsons[0]
        d = json.load(open(jp))
        catmap = {c["id"]: canon_class(c["name"]) for c in d["categories"]}
        unknown = {c["name"] for c in d["categories"] if canon_class(c["name"]) is None}
        if unknown:
            print("Albania categories ignored (not in the 5 classes):", unknown)
        anns = collections.defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        img_dir = os.path.dirname(jp)
        for im in d["images"]:
            p = os.path.join(img_dir, im["file_name"])
            if not os.path.exists(p):
                hits = glob.glob(os.path.join(src, "**", im["file_name"]), recursive=True)
                p = hits[0] if hits else p
            boxes = []
            for a in anns[im["id"]]:
                cls = catmap.get(a["category_id"])
                if cls is None:
                    continue
                x, y, w, h = a["bbox"]
                bb = _clip_box(x, y, x + w, y + h, im["width"], im["height"])
                if bb:
                    boxes.append((cls, *bb))
            records.append(dict(src=os.path.abspath(p), file_name=os.path.basename(p), country="Albania",
                                width=im["width"], height=im["height"], boxes=boxes))
    else:
        tmp_records, _, _ = scan_raw(src)
        for r in tmp_records:
            r["src"] = os.path.join(os.path.abspath(src), r["src"])
            r["country"] = "Albania"
        records = tmp_records
    coco = records_to_coco(records)
    with open(out_json, "w") as f:
        json.dump(coco, f)
    materialise_split(coco, "albania", "/")
    print("Albania:", count_table({"albania": coco})[0])
    return out_json


def ann_path(split):
    return os.path.join(C.DATA_ROOT, "annotations", f"{split}.json")


def img_dir(split):
    return os.path.join(C.DATA_ROOT, "images", split)
