"""
Green AI people detector: energy-aware two-tier YOLO26 pipeline.

Tier 1  YOLO26n (nano) runs on every frame.
Tier 2  YOLO26s (small) runs ONLY when tier 1 is uncertain, i.e. it finds a
        person with confidence inside the "uncertain band" (default 0.25-0.50).
CodeCarbon measures the energy (kWh) and estimated CO2 of the whole run.

Usage examples
    python green_detector.py                       # webcam, adaptive mode
    python green_detector.py --mode nano           # nano only
    python green_detector.py --mode small          # small only (baseline)
    python green_detector.py --source video.mp4
    python green_detector.py --source images/ --no-show --save out/
Press q to quit the live window.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
from codecarbon import EmissionsTracker
from ultralytics import YOLO

PERSON = 0  # COCO class id for "person"


class GreenDetector:
    """Two-tier detector. detect() returns (boxes, confs, tier_used)."""

    def __init__(self, mode="adaptive", nano="yolo26n.pt", small="yolo26s.pt",
                 conf=0.25, band=(0.25, 0.50), imgsz=640):
        self.mode, self.conf, self.band, self.imgsz = mode, conf, band, imgsz
        self.nano = YOLO(nano) if mode in ("nano", "adaptive") else None
        self.small = YOLO(small) if mode in ("small", "adaptive") else None
        self.frames = 0
        self.escalations = 0

    def _run(self, model, frame, conf):
        r = model.predict(frame, classes=[PERSON], conf=conf,
                          imgsz=self.imgsz, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    def is_uncertain(self, confs):
        lo, hi = self.band
        return bool(((confs >= lo) & (confs < hi)).any())

    def detect(self, frame):
        self.frames += 1
        if self.mode == "small":
            return (*self._run(self.small, frame, self.conf), "small")
        boxes, confs = self._run(self.nano, frame, self.conf)
        if self.mode == "adaptive" and self.is_uncertain(confs):
            self.escalations += 1
            return (*self._run(self.small, frame, self.conf), "small")
        return boxes, confs, "nano"


def draw(frame, boxes, confs, tier, fps):
    colour = (60, 180, 75) if tier == "nano" else (0, 140, 255)
    for (x1, y1, x2, y2), c in zip(boxes.astype(int), confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"person {c:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    hud = f"tier: YOLO26{'n' if tier == 'nano' else 's'} | people: {len(boxes)} | {fps:.1f} FPS"
    cv2.rectangle(frame, (0, 0), (len(hud) * 9 + 10, 26), (0, 0, 0), -1)
    cv2.putText(frame, hud, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def frames_from(source):
    """Yield BGR frames from a webcam index, video file, image file or folder."""
    p = Path(str(source))
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                yield cv2.imread(str(f))
        return
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        yield cv2.imread(str(p))
        return
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index, video, image or folder")
    ap.add_argument("--mode", choices=["adaptive", "nano", "small"], default="adaptive")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--band", type=float, nargs=2, default=(0.25, 0.50),
                    metavar=("LOW", "HIGH"), help="confidence band that triggers escalation")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--save", default="", help="folder to save annotated frames")
    args = ap.parse_args()

    det = GreenDetector(args.mode, conf=args.conf, band=tuple(args.band))
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(project_name=f"green-detect-{args.mode}",
                               measure_power_secs=1, log_level="error",
                               save_to_file=False)
    tracker.start()
    t0, fps = time.perf_counter(), 0.0
    try:
        for i, frame in enumerate(frames_from(args.source)):
            t = time.perf_counter()
            boxes, confs, tier = det.detect(frame)
            fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t, 1e-6) if i else 1 / max(time.perf_counter() - t, 1e-6)
            out = draw(frame, boxes, confs, tier, fps)
            if save_dir:
                cv2.imwrite(str(save_dir / f"frame_{i:05d}.jpg"), out)
            if not args.no_show:
                cv2.imshow("Green AI people detector", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and i + 1 >= args.max_frames:
                break
    finally:
        tracker.stop()
        cv2.destroyAllWindows()

    data = tracker.final_emissions_data
    secs = time.perf_counter() - t0
    n = max(det.frames, 1)
    joules = data.energy_consumed * 3.6e6
    summary = {
        "mode": args.mode,
        "frames": det.frames,
        "escalated_frames": det.escalations,
        "escalation_rate_%": round(100 * det.escalations / n, 1),
        "avg_fps": round(det.frames / secs, 2),
        "energy_kWh": f"{data.energy_consumed:.6f}",
        "energy_per_frame_J": round(joules / n, 3),
        "co2_g": f"{data.emissions * 1000:.6f}",
        "co2_per_1000_frames_g": round(data.emissions * 1e6 / n, 4),
    }
    print("\n=== Green AI run summary ===")
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    log = Path("run_log.csv")
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary.keys())
        if new:
            w.writeheader()
        w.writerow(summary)


if __name__ == "__main__":
    main()
