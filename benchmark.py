"""
Benchmark YOLO26s, YOLO26n and adaptive two-tier detector configurations.

Measures person detection accuracy, speed and CodeCarbon energy on
the same COCO128 images.

Usage:
    python benchmark.py
    python benchmark.py --data coco128.yaml --repeats 2

Outputs:
    benchmark_results.csv
    accuracy_vs_energy.png
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.utils import check_det_dataset

from green_detector import GreenDetector, PERSON, tracker


CONFIGS = [
    ("YOLO26s only", dict(mode="small")),
    ("YOLO26n only", dict(mode="nano")),
    ("Two-tier: band", dict(mode="adaptive", candidate=0)),
    ("Two-tier: band + candidates",
     dict(mode="adaptive", candidate=0.10)),
    ("Two-tier: band + candidates + scene",
     dict(mode="adaptive", candidate=0.10, scene=True)),
]


def load_gt(img_path):
    """Load YOLO person labels and return bounding boxes in xyxy pixels."""

    parts = list(Path(img_path).parts)

    image_idx = next(
        (i for i, part in enumerate(parts) if part.lower() == "images"),
        None
    )

    if image_idx is None:
        raise ValueError(
            f"Cannot find 'images' directory in: {img_path}"
        )

    # Replace images with labels in a platform-independent way.
    parts[image_idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")

    image = cv2.imread(str(img_path))

    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")

    height, width = image.shape[:2]

    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=float)

    rows = np.loadtxt(label_path, ndmin=2)

    # Keep person class only.
    rows = rows[rows[:, 0] == PERSON]

    if len(rows) == 0:
        return np.zeros((0, 4), dtype=float)

    cx = rows[:, 1] * width
    cy = rows[:, 2] * height
    bw = rows[:, 3] * width
    bh = rows[:, 4] * height

    return np.stack(
        [
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2
        ],
        axis=1
    )


def iou(a, b):
    """Calculate pairwise intersection over union."""

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = (
        np.clip(x2 - x1, 0, None)
        * np.clip(y2 - y1, 0, None)
    )

    def area(boxes):
        return (
            (boxes[:, 2] - boxes[:, 0])
            * (boxes[:, 3] - boxes[:, 1])
        )

    return inter / (
        area(a)[:, None] + area(b)[None, :] - inter + 1e-9
    )


def match(preds, gts, thr, min_conf=0.0):
    """Greedy prediction matching."""

    scores = []
    tps = []

    for (boxes, confs), gt in zip(preds, gts):

        keep = confs >= min_conf
        boxes = boxes[keep]
        confs = confs[keep]

        order = np.argsort(-confs)

        boxes = boxes[order]
        confs = confs[order]

        used = np.zeros(len(gt), dtype=bool)

        if len(gt) and len(boxes):
            ious = iou(boxes, gt)
        else:
            ious = np.zeros((len(boxes), 0))

        for i in range(len(boxes)):

            available = np.flatnonzero(~used)

            if len(available):
                j = int(
                    available[np.argmax(ious[i, available])]
                )
                hit = ious[i, j] >= thr
            else:
                j, hit = -1, False

            if hit:
                used[j] = True

            scores.append(confs[i])
            tps.append(hit)

    return (
        np.array(scores),
        np.array(tps, dtype=bool),
        sum(len(g) for g in gts)
    )


def average_precision(preds, gts, thr):
    """Calculate 101-point interpolated AP."""

    scores, tps, n_gt = match(preds, gts, thr)

    if len(scores) == 0 or n_gt == 0:
        return 0.0

    order = np.argsort(-scores)

    tp = tps[order].astype(float)

    fp = 1 - tp

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)

    recall = tp_c / n_gt
    precision = tp_c / (tp_c + fp_c)

    precision = np.maximum.accumulate(
        precision[::-1]
    )[::-1]

    ap_values = []

    for r in np.linspace(0, 1, 101):

        valid = recall >= r

        if valid.any():
            ap_values.append(precision[valid].max())
        else:
            ap_values.append(0.0)

    return float(np.mean(ap_values))


def precision_recall(preds, gts, conf=0.25, thr=0.5):
    """Calculate precision and recall at a fixed confidence threshold."""

    _, tps, n_gt = match(
        preds,
        gts,
        thr,
        min_conf=conf
    )

    tp = int(tps.sum())

    precision = tp / max(len(tps), 1)
    recall = tp / max(n_gt, 1)

    return precision, recall


def run(name, kwargs, images, gts, band):
    """Run one detector configuration and collect performance metrics."""

    # Measure model loading.
    t = tracker(f"load-{name}")
    t.start()

    t0 = time.perf_counter()

    det = GreenDetector(
        conf=0.001,
        band=band,
        every=0,
        **kwargs
    )

    load_s = time.perf_counter() - t0

    t.stop()

    load_j = (
        t.final_emissions_data.energy_consumed * 3.6e6
    )

    # Warm-up is excluded from the benchmark.
    warmup_image = cv2.imread(str(images[0]))

    if warmup_image is None:
        raise ValueError(f"Cannot read image: {images[0]}")

    det.detect(warmup_image)
    det.reset_stats()

    # Load benchmark images.
    frames = []

    for path in images:
        frame = cv2.imread(str(path))

        if frame is None:
            raise ValueError(f"Cannot read image: {path}")

        frames.append(frame)

    # Measure inference and energy.
    t = tracker(f"bench-{name}")

    preds = []

    t.start()
    t0 = time.perf_counter()

    for frame in frames:

        boxes, confs, _, _ = det.detect(frame)

        preds.append((boxes, confs))

    secs = time.perf_counter() - t0

    t.stop()

    data = t.final_emissions_data

    n = len(frames)

    precision, recall = precision_recall(
        preds,
        gts
    )

    return {
        "configuration": name,
        "images": n,

        "AP50": round(
            average_precision(preds, gts, 0.5),
            3
        ),

        "AP50-95": round(
            float(
                np.mean([
                    average_precision(preds, gts, threshold)
                    for threshold in np.arange(0.5, 0.96, 0.05)
                ])
            ),
            3
        ),

        "precision@0.25": round(precision, 3),
        "recall@0.25": round(recall, 3),

        "inference_ms": round(
            1000 * det.infer_s / n,
            1
        ),

        "end_to_end_ms": round(
            1000 * secs / n,
            1
        ),

        "energy_per_image_J": round(
            data.energy_consumed * 3.6e6 / n,
            3
        ),

        "total_energy_J": round(
            data.energy_consumed * 3.6e6,
            1
        ),

        "run_seconds": round(secs, 1),

        "escalation_%": round(
            100 * det.escalations / n,
            1
        ),

        "reasons": ";".join(
            f"{key}={value}"
            for key, value in det.reasons.items()
        ) or "-",

        "model_load_s": round(load_s, 2),
        "model_load_energy_J": round(load_j, 2),
    }


def plot(rows, path="accuracy_vs_energy.png"):
    """Create accuracy versus energy comparison plot."""

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(7, 4.2),
        dpi=180
    )

    colours = [
        "#e65100",
        "#2e7d32",
        "#1565c0",
        "#6a1b9a",
        "#00838f"
    ]

    for result, colour in zip(rows, colours):

        ax.scatter(
            result["energy_per_image_J"],
            result["AP50-95"],
            s=90,
            color=colour,
            label=result["configuration"],
            zorder=3
        )

    ax.set_xlabel(
        "Energy per image (J) - lower is greener"
    )

    ax.set_ylabel(
        "Person AP50-95 - higher is better"
    )

    ax.set_title(
        "Accuracy vs energy on the same test images"
    )

    ax.grid(alpha=0.3)

    ax.legend(
        fontsize=8,
        loc="lower right"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        facecolor="white"
    )

    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="coco128.yaml"
    )

    parser.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=(0.25, 0.50)
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat image set for steadier energy measurements"
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    # Load dataset information.
    info = check_det_dataset(args.data)

    img_dir = (
        Path(info["val"])
        if Path(info["val"]).is_dir()
        else Path(info["path"]) / info["val"]
    )

    images = sorted(
        path
        for path in img_dir.iterdir()
        if path.suffix.lower() in (".jpg", ".png", ".jpeg")
    )

    if not images:
        raise RuntimeError(
            f"No images found in dataset directory: {img_dir}"
        )

    # Load ground-truth annotations.
    gts = [
        load_gt(image_path)
        for image_path in images
    ]

    # Validate the dataset before benchmarking.
    person_boxes = sum(
        len(boxes)
        for boxes in gts
    )

    print(
        f"Dataset: {img_dir}",
        flush=True
    )

    print(
        f"Original images: {len(images)}",
        flush=True
    )

    print(
        f"Person ground-truth boxes: {person_boxes}",
        flush=True
    )

    if person_boxes == 0:
        raise RuntimeError(
            "No person annotations were loaded. "
            "Benchmark stopped to prevent invalid accuracy results."
        )

    # Repeat the image set if requested.
    images = images * args.repeats
    gts = gts * args.repeats

    print(
        f"\nBenchmark images: {len(images)}",
        flush=True
    )

    print(
        f"Benchmark person boxes: {sum(len(g) for g in gts)}",
        flush=True
    )

    rows = []

    # Run all configurations.
    for name, kwargs in CONFIGS:

        print(
            f"\nRunning configuration: {name}",
            flush=True
        )

        result = run(
            name,
            kwargs,
            images,
            gts,
            tuple(args.band)
        )

        rows.append(result)

        print(
            result,
            flush=True
        )

    # Save CSV results.
    output_csv = "benchmark_results.csv"

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    # Save comparison plot.
    plot(rows)

    print("\nBenchmark completed.", flush=True)
    print(f"Results saved to: {output_csv}", flush=True)
    print("Plot saved to: accuracy_vs_energy.png", flush=True)


if __name__ == "__main__":
    main()
