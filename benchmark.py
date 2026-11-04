"""
Benchmark YOLO26n, YOLO26s and the adaptive two-tier detector on the same
images: person-class accuracy (AP50, AP50-95), speed and CodeCarbon energy.

    python benchmark.py                      # uses COCO128 (downloads ~7 MB)
    python benchmark.py --data coco128.yaml --band 0.25 0.5

Results are written to benchmark_results.csv.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from codecarbon import EmissionsTracker
from ultralytics.data.utils import check_det_dataset

from green_detector import GreenDetector, PERSON


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


def average_precision(preds, gts, thr):
    """COCO-style 101-point AP for one class at one IoU threshold."""
    n_gt = sum(len(g) for g in gts)
    scores, tps = [], []
    for (boxes, confs), gt in zip(preds, gts):
        order = np.argsort(-confs)
        boxes, confs = boxes[order], confs[order]
        used = np.zeros(len(gt), bool)
        ious = iou(boxes, gt) if len(gt) and len(boxes) else np.zeros((len(boxes), 0))
        for i in range(len(boxes)):
            j = int(np.argmax(ious[i])) if ious.shape[1] else -1
            hit = j >= 0 and ious[i, j] >= thr and not used[j]
            if hit:
                used[j] = True
            scores.append(confs[i]); tps.append(hit)
    if not scores or n_gt == 0:
        return 0.0
    order = np.argsort(-np.array(scores))
    tp = np.array(tps)[order].astype(float)
    tp_c, fp_c = np.cumsum(tp), np.cumsum(1 - tp)
    recall, precision = tp_c / n_gt, tp_c / (tp_c + fp_c)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0
                          for r in np.linspace(0, 1, 101)]))


def run(mode, images, gts, band, conf):
    det = GreenDetector(mode, conf=conf, band=band)
    det.detect(cv2.imread(str(images[0])))           # warm-up (not measured)
    det.frames = det.escalations = 0
    frames = [cv2.imread(str(p)) for p in images]
    tracker = EmissionsTracker(project_name=f"bench-{mode}", measure_power_secs=1,
                               log_level="error", save_to_file=False)
    preds = []
    tracker.start(); t0 = time.perf_counter()
    for f in frames:
        b, c, _ = det.detect(f)
        preds.append((b, c))
    secs = time.perf_counter() - t0; tracker.stop()
    d = tracker.final_emissions_data
    n = len(frames)
    ap50 = average_precision(preds, gts, 0.5)
    ap = np.mean([average_precision(preds, gts, t) for t in np.arange(0.5, 0.96, 0.05)])
    return {
        "mode": mode, "images": n,
        "AP50_person": round(ap50, 3), "AP50-95_person": round(float(ap), 3),
        "ms_per_image": round(1000 * secs / n, 1),
        "energy_per_image_J": round(d.energy_consumed * 3.6e6 / n, 3),
        "co2_per_1000_images_g": round(d.emissions * 1e6 / n, 4),
        "escalation_rate_%": round(100 * det.escalations / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coco128.yaml")
    ap.add_argument("--band", type=float, nargs=2, default=(0.25, 0.50))
    ap.add_argument("--conf", type=float, default=0.001, help="low conf for fair AP")
    ap.add_argument("--repeats", type=int, default=1, help="repeat image set for steadier energy")
    args = ap.parse_args()

    info = check_det_dataset(args.data)
    img_dir = Path(info["val"]) if Path(info["val"]).is_dir() else Path(info["path"]) / info["val"]
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    gts = [load_gt(p) for p in images]
    images, gts = images * args.repeats, gts * args.repeats
    print(f"{len(images)} images, {sum(len(g) for g in gts)} person boxes")

    rows = [run(m, images, gts, tuple(args.band), args.conf) for m in ("small", "nano", "adaptive")]
    with open("benchmark_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
