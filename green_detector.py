"""
Green AI People Detector: adaptive, energy-aware YOLO26 inference.

YOLO26n runs on every frame. YOLO26s is invoked when detections are uncertain,
the scene is difficult/crowded, or a periodic audit is due. Scene heuristics and
periodic audits help address confidence-only routing's inability to escalate
when the nano model completely misses a person.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from codecarbon import EmissionsTracker
from ultralytics import YOLO

PERSON = 0


def analyse_scene(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


class GreenDetector:
    def __init__(
        self, mode="adaptive", nano="yolo26n.pt", small="yolo26s.pt",
        conf=0.25, band=(0.25, 0.50), imgsz=640, scene_aware=True,
        audit_every=30, blur_threshold=60.0, dark_threshold=55.0,
        bright_threshold=220.0, contrast_threshold=22.0, crowd_threshold=5,
    ):
        self.mode, self.conf, self.band, self.imgsz = mode, conf, band, imgsz
        self.scene_aware = scene_aware
        self.audit_every = max(0, int(audit_every))
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.contrast_threshold = contrast_threshold
        self.crowd_threshold = crowd_threshold
        self.nano = YOLO(nano) if mode in ("nano", "adaptive") else None
        self.small = YOLO(small) if mode in ("small", "adaptive") else None
        self.frames = 0
        self.escalations = 0
        self.last_reasons = []
        self.tier_counts = {"nano": 0, "small": 0}

    def _run(self, model, frame, conf):
        result = model.predict(
            frame, classes=[PERSON], conf=conf, imgsz=self.imgsz, verbose=False
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4), dtype=float), np.empty((0,), dtype=float)
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()

    def is_uncertain(self, confidences):
        low, high = self.band
        return bool(((confidences >= low) & (confidences < high)).any())

    def escalation_reasons(self, frame, boxes, confidences):
        reasons = []
        if self.is_uncertain(confidences):
            reasons.append("uncertain_detection")
        if self.crowd_threshold > 0 and len(boxes) >= self.crowd_threshold:
            reasons.append("crowded_scene")

        if self.scene_aware:
            scene = analyse_scene(frame)
            if scene["brightness"] < self.dark_threshold:
                reasons.append("low_light")
            if scene["brightness"] > self.bright_threshold:
                reasons.append("overexposed")
            if scene["blur_score"] < self.blur_threshold:
                reasons.append("motion_blur_or_low_detail")
            if scene["contrast"] < self.contrast_threshold:
                reasons.append("low_contrast")

        if self.audit_every and self.frames % self.audit_every == 0:
            reasons.append("periodic_miss_audit")
        return reasons

    def detect(self, frame):
        self.frames += 1
        self.last_reasons = []

        if self.mode == "small":
            boxes, scores = self._run(self.small, frame, self.conf)
            self.tier_counts["small"] += 1
            return boxes, scores, "small"

        boxes, scores = self._run(self.nano, frame, self.conf)
        if self.mode == "adaptive":
            self.last_reasons = self.escalation_reasons(frame, boxes, scores)
            if self.last_reasons:
                self.escalations += 1
                boxes, scores = self._run(self.small, frame, self.conf)
                self.tier_counts["small"] += 1
                return boxes, scores, "small"

        self.tier_counts["nano"] += 1
        return boxes, scores, "nano"


def draw(frame, boxes, confidences, tier, fps, reasons=None):
    colour = (60, 180, 75) if tier == "nano" else (0, 140, 255)
    for (x1, y1, x2, y2), score in zip(boxes.astype(int), confidences):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"person {score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    model_label = "YOLO26n" if tier == "nano" else "YOLO26s"
    hud = f"{model_label} | people: {len(boxes)} | {fps:.1f} FPS"
    cv2.rectangle(frame, (0, 0), (min(len(hud) * 10 + 12, frame.shape[1]), 28),
                  (0, 0, 0), -1)
    cv2.putText(frame, hud, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    if reasons:
        cv2.putText(frame, ("Escalated: " + ", ".join(reasons))[:100], (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
    return frame


def frames_from(source):
    path = Path(str(source))
    if path.is_dir():
        for item in sorted(path.iterdir()):
            if item.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                frame = cv2.imread(str(item))
                if frame is not None:
                    yield frame
        return
    if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        frame = cv2.imread(str(path))
        if frame is not None:
            yield frame
        return

    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="webcam index, video, image or folder")
    parser.add_argument("--mode", choices=["adaptive", "nano", "small"], default="adaptive")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--band", type=float, nargs=2, default=(0.25, 0.50),
                        metavar=("LOW", "HIGH"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save", default="")
    parser.add_argument("--audit-every", type=int, default=30,
                        help="escalate every N frames to audit possible nano misses; 0 disables")
    parser.add_argument("--no-scene-aware", action="store_true")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if not 0 <= args.band[0] < args.band[1] <= 1:
        parser.error("--band must satisfy 0 <= LOW < HIGH <= 1")

    detector = GreenDetector(
        mode=args.mode, conf=args.conf, band=tuple(args.band), imgsz=args.imgsz,
        scene_aware=not args.no_scene_aware, audit_every=args.audit_every,
    )
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=f"green-people-detector-{args.mode}",
        measure_power_secs=1, log_level="error", save_to_file=False,
    )
    tracker.start()
    start_time = time.perf_counter()
    total_latency = 0.0

    try:
        for index, frame in enumerate(frames_from(args.source)):
            inference_start = time.perf_counter()
            boxes, scores, tier = detector.detect(frame)
            latency = time.perf_counter() - inference_start
            total_latency += latency
            instant_fps = 1.0 / latency if latency > 0 else 0.0
            annotated = draw(frame, boxes, scores, tier, instant_fps,
                             detector.last_reasons)

            if save_dir:
                cv2.imwrite(str(save_dir / f"frame_{index:05d}.jpg"), annotated)
            if not args.no_show:
                cv2.imshow("Green AI People Detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and index + 1 >= args.max_frames:
                break
    finally:
        tracker.stop()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    count = max(detector.frames, 1)
    emissions = tracker.final_emissions_data
    energy_kwh = float(emissions.energy_consumed)
    co2_kg = float(emissions.emissions)

    summary = {
        "mode": args.mode,
        "frames": detector.frames,
        "nano_frames": detector.tier_counts["nano"],
        "small_frames": detector.tier_counts["small"],
        "escalated_frames": detector.escalations,
        "escalation_rate_pct": round(100 * detector.escalations / count, 2),
        "wall_clock_fps": round(detector.frames / elapsed, 2) if elapsed else 0,
        "mean_detector_latency_ms": round(1000 * total_latency / count, 2),
        "energy_kwh": round(energy_kwh, 9),
        "energy_per_frame_j": round(energy_kwh * 3_600_000 / count, 6),
        "co2_kg": round(co2_kg, 9),
        "co2_g_per_1000_frames": round(co2_kg * 1_000_000 / count, 6),
    }

    print("\n=== Green AI run summary ===")
    for key, value in summary.items():
        print(f"{key:>30}: {value}")

    log_path = Path("run_log.csv")
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
