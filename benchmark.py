"""
Compare YOLO26n, YOLO26s and adaptive routing on the same labelled images.

Example:
    python benchmark.py --data coco128.yaml --repeats 5

The person AP values are a lightweight 101-point approximation, not an
official COCO evaluation. Use Ultralytics validation/pycocotools for formal
benchmark reporting.
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


def load_ground_truth(image_path):
    label_path = Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    if not label_path.exists():
        return np.zeros((0, 4), dtype=float)

    rows = np.loadtxt(label_path, ndmin=2)
    rows = rows[rows[:, 0] == PERSON]
    if len(rows) == 0:
        return np.zeros((0, 4), dtype=float)

    cx, cy = rows[:, 1] * width, rows[:, 2] * height
    bw, bh = rows[:, 3] * width, rows[:, 4] * height
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def calculate_iou(predicted, ground_truth):
    if len(predicted) == 0 or len(ground_truth) == 0:
        return np.zeros((len(predicted), len(ground_truth)), dtype=float)
    x1 = np.maximum(predicted[:, None, 0], ground_truth[None, :, 0])
    y1 = np.maximum(predicted[:, None, 1], ground_truth[None, :, 1])
    x2 = np.minimum(predicted[:, None, 2], ground_truth[None, :, 2])
    y2 = np.minimum(predicted[:, None, 3], ground_truth[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    pred_area = (predicted[:, 2] - predicted[:, 0]) * (predicted[:, 3] - predicted[:, 1])
    gt_area = (ground_truth[:, 2] - ground_truth[:, 0]) * (ground_truth[:, 3] - ground_truth[:, 1])
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return intersection / (union + 1e-9)


def average_precision(predictions, ground_truths, iou_threshold):
    total_gt = sum(len(boxes) for boxes in ground_truths)
    if total_gt == 0:
        return float("nan")

    detections = []
    for (boxes, scores), gt_boxes in zip(predictions, ground_truths):
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        overlaps = calculate_iou(boxes, gt_boxes)
        matched = np.zeros(len(gt_boxes), dtype=bool)

        for index, score in enumerate(scores):
            best_gt = int(np.argmax(overlaps[index])) if len(gt_boxes) else -1
            hit = (best_gt >= 0 and overlaps[index, best_gt] >= iou_threshold
                   and not matched[best_gt])
            if hit:
                matched[best_gt] = True
            detections.append((float(score), int(hit)))

    if not detections:
        return 0.0

    detections.sort(key=lambda item: item[0], reverse=True)
    tp = np.cumsum([item[1] for item in detections])
    fp = np.cumsum([1 - item[1] for item in detections])
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

    samples = [
        precision_envelope[recall >= level].max()
        if np.any(recall >= level) else 0.0
        for level in np.linspace(0, 1, 101)
    ]
    return float(np.mean(samples))


def run_benchmark(mode, image_paths, ground_truths, band, confidence):
    detector = GreenDetector(
        mode=mode, conf=confidence, band=band,
        scene_aware=(mode == "adaptive"), audit_every=30,
    )

    # Warm up once, then reset counters before the measured run.
    detector.detect(cv2.imread(str(image_paths[0])))
    detector.frames = detector.escalations = 0
    detector.tier_counts = {"nano": 0, "small": 0}

    tracker = EmissionsTracker(
        project_name=f"green-ai-benchmark-{mode}",
        measure_power_secs=1, log_level="error", save_to_file=False,
    )
    predictions = []
    total_latency = 0.0
    tracker.start()
    start = time.perf_counter()
    try:
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            inference_start = time.perf_counter()
            boxes, scores, _ = detector.detect(image)
            total_latency += time.perf_counter() - inference_start
            predictions.append((boxes, scores))
    finally:
        tracker.stop()

    elapsed = time.perf_counter() - start
    emissions = tracker.final_emissions_data
    count = max(len(image_paths), 1)
    ap50 = average_precision(predictions, ground_truths, 0.50)
    ap_values = [
        average_precision(predictions, ground_truths, threshold)
        for threshold in np.arange(0.50, 0.96, 0.05)
    ]
    ap5095 = float(np.nanmean(ap_values))

    energy_kwh = float(emissions.energy_consumed)
    co2_kg = float(emissions.emissions)
    return {
        "mode": mode,
        "images": len(image_paths),
        "person_AP50_approx": round(ap50, 4),
        "person_AP50_95_approx": round(ap5095, 4),
        "mean_latency_ms": round(1000 * total_latency / count, 2),
        "wall_clock_fps": round(count / elapsed, 2) if elapsed else 0,
        "energy_kwh": round(energy_kwh, 9),
        "energy_per_image_j": round(energy_kwh * 3_600_000 / count, 6),
        "co2_kg": round(co2_kg, 9),
        "co2_g_per_1000_images": round(co2_kg * 1_000_000 / count, 6),
        "escalation_rate_pct": round(100 * detector.escalations / count, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument("--band", type=float, nargs=2, default=(0.25, 0.50))
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="benchmark_results.csv")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not 0 <= args.band[0] < args.band[1] <= 1:
        parser.error("--band must satisfy 0 <= LOW < HIGH <= 1")

    dataset = check_det_dataset(args.data)
    image_dir = Path(dataset["val"])
    if not image_dir.is_dir():
        image_dir = Path(dataset["path"]) / dataset["val"]
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise SystemExit(f"No validation images found in {image_dir}")

    ground_truths = [load_ground_truth(path) for path in image_paths]
    image_paths *= args.repeats
    ground_truths *= args.repeats
    print(f"{len(image_paths)} image passes; {sum(map(len, ground_truths))} person annotations.")

    rows = [
        run_benchmark(mode, image_paths, ground_truths, tuple(args.band), args.conf)
        for mode in ("small", "nano", "adaptive")
    ]
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nBenchmark results (AP values are approximations):")
    for row in rows:
        print(row)
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
