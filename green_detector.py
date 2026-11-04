"""
Green AI people detector: energy-aware two-tier YOLO26 pipeline.

Tier 1  YOLO26n (nano) runs on every frame.
Tier 2  YOLO26s (small) runs ONLY when the frame looks unreliable. Triggers:
          uncertain  - a person detected with confidence in [0.25, 0.50)
          candidate  - a weak person candidate in [0.10, 0.25), i.e. a person
                       YOLO26n may have missed
          scene      - the image is very dark or blurred (optional, --scene)
          periodic   - every N frames, as a safety check on video (--every)
CodeCarbon measures model-loading energy and inference energy separately.

Usage examples
    python green_detector.py                       # webcam, adaptive mode
    python green_detector.py --mode nano           # nano only
    python green_detector.py --mode small          # small only (baseline)
    python green_detector.py --source video.mp4 --scene --every 30
    python green_detector.py --source images/ --no-show --save out/
Press q to quit the live window.
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from codecarbon import EmissionsTracker
from ultralytics import YOLO

PERSON = 0  # COCO class id for "person"


def scene_quality(frame):
    """Return (brightness 0-255, sharpness) from a small greyscale copy."""
    g = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
    return float(g.mean()), float(cv2.Laplacian(g, cv2.CV_64F).var())


class GreenDetector:
    """Two-tier detector. detect() returns (boxes, confs, tier, reason)."""

    def __init__(self, mode="adaptive", nano="yolo26n.pt", small="yolo26s.pt", conf=0.25,
                 band=(0.25, 0.50), candidate=0.10, scene=False, dark=40.0, blur=30.0,
                 every=0, imgsz=640):
        self.mode, self.conf, self.band, self.imgsz = mode, conf, band, imgsz
        self.candidate, self.scene, self.dark, self.blur, self.every = candidate, scene, dark, blur, every
        self.nano = YOLO(nano) if mode in ("nano", "adaptive") else None
        self.small = YOLO(small) if mode in ("small", "adaptive") else None
        self.reset_stats()

    def reset_stats(self):
        self.frames = 0
        self.reasons = Counter()        # why frames were escalated
        self.infer_s = 0.0              # time spent inside the models only

    @property
    def escalations(self):
        return sum(self.reasons.values())

    def _run(self, model, frame, conf):
        t = time.perf_counter()
        r = model.predict(frame, classes=[PERSON], conf=conf, imgsz=self.imgsz, verbose=False)[0]
        self.infer_s += time.perf_counter() - t
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    def escalation_reason(self, frame, confs):
        """Return why this frame needs YOLO26s, or None if YOLO26n is trusted."""
        lo, hi = self.band
        if ((confs >= lo) & (confs < hi)).any():
            return "uncertain"
        if self.candidate and ((confs >= self.candidate) & (confs < lo)).any():
            return "candidate"
        if self.scene:
            bright, sharp = scene_quality(frame)
            if bright < self.dark or sharp < self.blur:
                return "scene"
        if self.every and self.frames % self.every == 0:
            return "periodic"
        return None

    def detect(self, frame):
        self.frames += 1
        if self.mode == "small":
            return (*self._run(self.small, frame, self.conf), "small", None)
        # run nano low enough to see weak candidates, then filter for output
        low = min(self.conf, self.candidate or self.conf)
        boxes, confs = self._run(self.nano, frame, low)
        if self.mode == "adaptive":
            reason = self.escalation_reason(frame, confs)
            if reason:
                self.reasons[reason] += 1
                return (*self._run(self.small, frame, self.conf), "small", reason)
        keep = confs >= self.conf
        return boxes[keep], confs[keep], "nano", None


def draw(frame, boxes, confs, tier, reason, fps):
    colour = (60, 180, 75) if tier == "nano" else (0, 140, 255)
    for (x1, y1, x2, y2), c in zip(boxes.astype(int), confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"person {c:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    hud = f"YOLO26{'n' if tier == 'nano' else 's'}" + (f" ({reason})" if reason else "")
    hud += f" | people: {len(boxes)} | {fps:.1f} FPS"
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


def tracker(name):
    return EmissionsTracker(project_name=name, measure_power_secs=1, log_level="error",
                            save_to_file=False, allow_multiple_runs=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index, video, image or folder")
    ap.add_argument("--mode", choices=["adaptive", "nano", "small"], default="adaptive")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--band", type=float, nargs=2, default=(0.25, 0.50), metavar=("LOW", "HIGH"),
                    help="confidence band that triggers escalation")
    ap.add_argument("--candidate", type=float, default=0.10,
                    help="weak candidates at or above this (below LOW) also escalate; 0 disables")
    ap.add_argument("--scene", action="store_true", help="escalate dark or blurred frames")
    ap.add_argument("--every", type=int, default=30, help="periodic escalation every N frames (0 = off)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--save", default="", help="folder to save annotated frames")
    args = ap.parse_args()

    load_t = tracker("load"); load_t.start(); t_load = time.perf_counter()
    det = GreenDetector(args.mode, conf=args.conf, band=tuple(args.band), candidate=args.candidate,
                        scene=args.scene, every=args.every)
    load_secs = time.perf_counter() - t_load; load_t.stop()
    load_kwh = load_t.final_emissions_data.energy_consumed

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    run_t = tracker(f"green-detect-{args.mode}"); run_t.start()
    t0, fps = time.perf_counter(), 0.0
    try:
        for i, frame in enumerate(frames_from(args.source)):
            t = time.perf_counter()
            boxes, confs, tier, reason = det.detect(frame)
            dt = max(time.perf_counter() - t, 1e-6)
            fps = 1 / dt if i == 0 else 0.9 * fps + 0.1 / dt
            out = draw(frame, boxes, confs, tier, reason, fps)
            if save_dir:
                cv2.imwrite(str(save_dir / f"frame_{i:05d}.jpg"), out)
            if not args.no_show:
                cv2.imshow("Green AI people detector", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and i + 1 >= args.max_frames:
                break
    finally:
        run_t.stop()
        cv2.destroyAllWindows()

    d = run_t.final_emissions_data
    secs = time.perf_counter() - t0
    n = max(det.frames, 1)
    summary = {
        "mode": args.mode,
        "frames": det.frames,
        "escalated_frames": det.escalations,
        "escalation_rate_%": round(100 * det.escalations / n, 1),
        "reasons": ";".join(f"{k}={v}" for k, v in det.reasons.items()) or "-",
        "inference_fps": round(det.frames / max(det.infer_s, 1e-6), 2),
        "end_to_end_fps": round(det.frames / secs, 2),
        "run_seconds": round(secs, 1),
        "run_energy_kWh": f"{d.energy_consumed:.6f}",
        "energy_per_frame_J": round(d.energy_consumed * 3.6e6 / n, 3),
        "co2_g": f"{d.emissions * 1000:.6f}",
        "co2_per_1000_frames_g": round(d.emissions * 1e6 / n, 4),
        "model_load_s": round(load_secs, 2),
        "model_load_energy_J": round(load_kwh * 3.6e6, 2),
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
