# Green AI People Detector — Energy-Aware Adaptive YOLO26

A practical CO3091/7091 prototype comparing a lightweight YOLO26 nano detector, a larger YOLO26 small detector and an adaptive two-tier strategy. CodeCarbon records run-level energy and estimated CO2 emissions.

## Research question

Can adaptive inference reduce energy use while retaining useful person-detection performance compared with always using a larger detector?

## System design

1. YOLO26n runs on every frame as the lightweight first stage.
2. Adaptive routing escalates selected frames when a detection is uncertain, the scene is crowded or visually difficult, or a periodic audit is due.
3. YOLO26s rechecks escalated frames.
4. CodeCarbon records estimated run-level energy and emissions.
5. The benchmark compares person detection quality, latency, throughput and energy on the same labelled images.

Scene-aware routing uses simple image statistics (brightness, contrast and Laplacian variance). These are heuristics, not a learned scene classifier. Periodic audits are included because confidence-only routing cannot escalate when the nano model returns no detection.

## Files

- green_detector.py — webcam, image, video and folder inference
- benchmark.py — comparison of nano, small and adaptive modes
- requirements.txt — Python dependencies
- benchmark_results.csv — existing repository benchmark record; verify its provenance before citing it
- run_log.csv — generated locally after running the detector

## Setup (Windows PowerShell)

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
venv\Scripts\activate.bat
```

## Run the demonstration

Webcam, adaptive mode:

```bash
python green_detector.py
```

Compare modes on the same webcam and conditions:

```bash
python green_detector.py --mode small
python green_detector.py --mode nano
python green_detector.py --mode adaptive
```

Run on video or a directory of images:

```bash
python green_detector.py --source sample.mp4 --mode adaptive --save outputs/video
python green_detector.py --source images/ --no-show --save outputs/images
```

Routing controls:

```bash
python green_detector.py --band 0.25 0.60
python green_detector.py --audit-every 15
python green_detector.py --no-scene-aware
```

The first run downloads YOLO26 model weights automatically.

## Reproducible benchmark

```bash
python benchmark.py --data coco128.yaml --repeats 5
```

This evaluates the same validation images with all three modes and writes benchmark_results.csv. The person AP values are a lightweight 101-point approximation, not an official COCO evaluation. Use the official Ultralytics validation pipeline and/or pycocotools for formal COCO mAP.

## Measurement guidance

- Use the same computer, images, image size, confidence threshold and software versions for every mode.
- Keep power mode, screen brightness, background applications and device temperature as consistent as possible.
- Repeat runs and report mean and variation rather than relying on one run.
- CodeCarbon estimates energy and emissions; accuracy depends on available hardware telemetry and regional emissions factors.
- Very short runs may be shorter than the energy sampling interval. Use a sufficiently long benchmark.
- Distinguish detector latency from end-to-end wall-clock FPS.
- Confidence scores are not accuracy measurements.
- Do not claim energy savings until repeated measurements support the claim.
- The existing CSV is historical repository data. Re-run the benchmark and record machine/software details before using those values in an academic report.

## Suggested report contribution

Compare person AP50 and AP50-95, mean detector latency, end-to-end throughput, energy per image/frame, CO2 emissions per 1,000 images and adaptive escalation rate. Discuss trade-offs and failure cases including missed people, false positives, occlusion, small/distant people, blur and difficult lighting.

The periodic audit and scene-quality rules are safeguards, not guarantees that every missed person will be recovered.

## References

- Schwartz, R. et al. (2020) 'Green AI', Communications of the ACM, 63(12), pp. 54–63. https://doi.org/10.1145/3381831
- Lin, T.-Y. et al. (2014) 'Microsoft COCO: Common Objects in Context', ECCV 2014. https://doi.org/10.1007/978-3-319-10602-1_48
- Ultralytics, YOLO26 documentation: https://docs.ultralytics.com/models/yolo26/
- CodeCarbon documentation: https://docs.codecarbon.io/

## Coursework integrity

Run the system yourself, retain your outputs and screenshots, and only report measurements you personally reproduce. Ensure the final academic report distinguishes implemented features from proposed future work.
