"""
Benchmark YOLO26s, YOLO26n and versions of the adaptive two-tier detector on the
same images: person accuracy (AP50, AP50-95, precision, recall), speed and
CodeCarbon energy. Each configuration is an ablation, so the effect of every
escalation trigger can be seen separately.

    python benchmark.py                      # uses COCO128 (downloads ~7 MB)
    python benchmark.py --data coco128.yaml --repeats 2

Writes benchmark_results.csv and accuracy_vs_energy.png.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.utils import check_det_dataset

from green_detector import GreenDetector, PERSON, tracker

CONFIGS = [  # name, GreenDetector kwargs
    ("YOLO26s only", dict(mode="small")),
    ("YOLO26n only", dict(mode="nano")),
    ("Two-tier: band", dict(mode="adaptive", candidate=0)),
    ("Two-tier: band + candidates", dict(mode="adaptive", candidate=0.10)),
    ("Two-tier: band + candidates + scene", dict(mode="adaptive", candidate=0.10, scene=True)),
]


def load_gt(img_path):
    """Read YOLO-format labels for one image -> person boxes in xyxy pixels."""
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    h, w = cv2.imread(str(img_path)).shape[:2]
    if not lbl.exists():
        return np.zeros((0, 4))
    rows = np.loadtxt(lbl, ndmin=2)
    rows = rows[rows[:, 0] == PERSON]
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(preds, gts, thr, min_conf=0.0):
    """Greedy matching -> (scores, true-positive flags, number of ground-truth boxes)."""
    scores, tps = [], []
    for (boxes, confs), gt in zip(preds, gts):
        keep = confs >= min_conf
        boxes, confs = boxes[keep], confs[keep]
        order = np.argsort(-confs)
        boxes, confs = boxes[order], confs[order]
        used = np.zeros(len(gt), bool)
        ious = iou(boxes, gt) if len(gt) and len(boxes) else np.zeros((len(boxes), 0))
        for i in range(len(boxes)):
            # Match each prediction to the best still-unmatched ground-truth box.
            # Choosing the global best first can incorrectly reject a valid
            # second-best match when the best box has already been claimed.
            available = np.flatnonzero(~used)
            if len(available):
                j = int(available[np.argmax(ious[i, available])])
                hit = ious[i, j] >= thr
            else:
                j, hit = -1, False
            if hit:
                used[j] = True
            scores.append(confs[i]); tps.append(hit)
    return np.array(scores), np.array(tps, bool), sum(len(g) for g in gts)


def average_precision(preds, gts, thr):
    """COCO-style 101-point AP for one class at one IoU threshold."""
    scores, tps, n_gt = match(preds, gts, thr)
    if len(scores) == 0 or n_gt == 0:
        return 0.0
    tp = tps[np.argsort(-scores)].astype(float)
    tp_c, fp_c = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = tp_c / n_gt, tp_c / (tp_c + fp_c)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0
                          for r in np.linspace(0, 1, 101)]))


def precision_recall(preds, gts, conf=0.25, thr=0.5):
    _, tps, n_gt = match(preds, gts, thr, min_conf=conf)
    tp = int(tps.sum())
    return tp / max(len(tps), 1), tp / max(n_gt, 1)


def run(name, kwargs, images, gts, band):
    t = tracker(f"load-{name}"); t.start(); t0 = time.perf_counter()
    det = GreenDetector(conf=0.001, band=band, every=0, **kwargs)  # every=0: images are independent
    load_s = time.perf_counter() - t0; t.stop()
    load_j = t.final_emissions_data.energy_consumed * 3.6e6

    det.detect(cv2.imread(str(images[0])))   # warm-up (not measured)
    det.reset_stats()
    frames = [cv2.imread(str(p)) for p in images]
    t = tracker(f"bench-{name}")
    preds = []
    t.start(); t0 = time.perf_counter()
    for f in frames:
        b, c, _, _ = det.detect(f)
        preds.append((b, c))
    secs = time.perf_counter() - t0; t.stop()
    d = t.final_emissions_data
    n = len(frames)
    p, r = precision_recall(preds, gts)
    return {
        "configuration": name, "images": n,
        "AP50": round(average_precision(preds, gts, 0.5), 3),
        "AP50-95": round(float(np.mean([average_precision(preds, gts, x) for x in np.arange(0.5, 0.96, 0.05)])), 3),
        "precision@0.25": round(p, 3), "recall@0.25": round(r, 3),
        "inference_ms": round(1000 * det.infer_s / n, 1),
        "end_to_end_ms": round(1000 * secs / n, 1),
        "energy_per_image_J": round(d.energy_consumed * 3.6e6 / n, 3),
        "total_energy_J": round(d.energy_consumed * 3.6e6, 1),
        "run_seconds": round(secs, 1),
        "escalation_%": round(100 * det.escalations / n, 1),
        "reasons": ";".join(f"{k}={v}" for k, v in det.reasons.items()) or "-",
        "model_load_s": round(load_s, 2), "model_load_energy_J": round(load_j, 2),
    }


def plot(rows, path="accuracy_vs_energy.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=180)
    colours = ["#e65100", "#2e7d32", "#1565c0", "#6a1b9a", "#00838f"]
    for r, c in zip(rows, colours):
        ax.scatter(r["energy_per_image_J"], r["AP50-95"], s=90, color=c, label=r["configuration"], zorder=3)
    ax.set_xlabel("Energy per image (J)  -  lower is greener")
    ax.set_ylabel("Person AP50-95  -  higher is better")
    ax.set_title("Accuracy vs energy on the same test images")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coco128.yaml")
    ap.add_argument("--band", type=float, nargs=2, default=(0.25, 0.50))
    ap.add_argument("--repeats", type=int, default=1, help="repeat image set for steadier energy")
    args = ap.parse_args()

    info = check_det_dataset(args.data)
    img_dir = Path(info["val"]) if Path(info["val"]).is_dir() else Path(info["path"]) / info["val"]
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    gts = [load_gt(p) for p in images]
    images, gts = images * args.repeats, gts * args.repeats
    print(f"{len(images)} images, {sum(len(g) for g in gts)} person boxes")

    rows = []
    for name, kw in CONFIGS:
        rows.append(run(name, kw, images, gts, tuple(args.band)))
        print(rows[-1], flush=True)
    with open("benchmark_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    plot(rows)


if __name__ == "__main__":
    main()
