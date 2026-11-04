# Green AI People Detector (YOLO26 + CodeCarbon)

An energy-aware, two-tier people detector for CO3091/7091. The research question is whether a lightweight-first detector can keep the accuracy of a larger model while using much less energy.

![System architecture](report/system_diagram.png)

## Real-time dashboard sample

![Green AI YOLO26 real-time people detection dashboard](docs/images/realtime_dashboard_sample.jpg)

*Illustrative dashboard prototype. FPS, people count, energy and CO2 values shown in the screenshot are sample values, not verified measurements from a live benchmark.*

## How it works

- **Tier 1: YOLO26n (nano, 2.4M params)** runs on every frame.
- **Tier 2: YOLO26s (small, 9.5M params)** re-processes a frame only when YOLO26n looks unreliable:

| Trigger | Fires when | Default |
|---|---|---|
| `uncertain` | a person is detected with confidence 0.25–0.50 | on |
| `candidate` | a weak candidate at 0.10–0.25, so a person may have been missed | on (`--candidate 0` disables) |
| `scene` | frame is very dark (mean brightness < 40) or blurred (Laplacian variance < 30) | off (`--scene` enables) |
| `periodic` | every N-th video frame, as a safety check | every 30 frames (`--every 0` disables) |

- **CodeCarbon** measures inference energy and model-loading energy separately.

On screen, green boxes mean YOLO26n handled the frame. Orange boxes mean the frame was escalated, and the display shows the reason.

## Results

These are person-class results on COCO128: 128 images with 254 people, run on a 2-vCPU CPU. Reproduce them with `python benchmark.py`.

| Configuration | AP50 | AP50–95 | Recall | ms/image | J/image | Escalated |
|---|---|---|---|---|---|---|
| YOLO26s only (baseline) | 0.826 | 0.615 | 0.740 | 210.5 | 2.558 | – |
| YOLO26n only | 0.776 | 0.539 | 0.661 | 82.9 | 0.972 | – |
| Two-tier: band only | 0.824 | 0.601 | 0.728 | 125.1 | 1.466 | 20.3% |
| **Two-tier: band + candidates (default)** | **0.822** | **0.601** | **0.740** | **133.6** | **1.566** | **25.0%** |
| Two-tier: band + candidates + scene | 0.822 | 0.602 | 0.740 | 147.9 | 1.735 | 27.3% |

![Accuracy vs energy](accuracy_vs_energy.png)

- The default two-tier detector **matches YOLO26s's recall** and keeps 99.5% of its AP50 while using **39% less energy** per image.
- The candidate trigger brings recall back from 0.728 to 0.740 for 7% more energy. This covers the case where YOLO26n misses a person entirely.
- Scene checks added cost with no gain on COCO128, which has few dark or blurred images, so they are optional.
- Loading both models costs about 0.13 s and 1.5 J, which is less than processing one image.

**Methodology:** every configuration runs on the same images and the same machine, with one warm-up image excluded. AP is computed COCO-style (101-point) for the person class. Recall is measured at confidence 0.25 and IoU 0.5. Periodic escalation is off in the benchmark because the test images are independent rather than video frames. Without hardware power counters, CodeCarbon estimates CPU power from the processor's TDP. That means the relative comparison is more reliable than the absolute joule values.

## Setup

Windows:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The model weights (`yolo26n.pt`, `yolo26s.pt`) download automatically on first run.

## Run

```
python green_detector.py                        # webcam, adaptive mode (press q to quit)
python green_detector.py --mode nano            # YOLO26n only
python green_detector.py --mode small           # YOLO26s only (baseline)
python green_detector.py --scene                # also escalate dark / blurred frames
python green_detector.py --source video.mp4 --every 15
python green_detector.py --source images/ --no-show --save out/
python benchmark.py                             # reproduce the results table and chart
```

Each run prints a summary and appends it to `run_log.csv`. The summary includes:
- escalation rate and the reason for each escalation
- inference FPS and end-to-end FPS
- energy per frame and CO₂e
- model-loading time and energy

## Demo plan for the TA (bonus marks)

1. Run `--mode small` on the webcam for about 30 s. This is the baseline.
2. Run `--mode nano` for about 30 s.
3. Run the default adaptive mode for about 30 s. Walk partly out of view or dim the lights to show escalations and their reasons on screen.
4. Open `run_log.csv` and compare energy per frame, FPS and escalation rate.

## Files

| File | Purpose |
|---|---|
| `green_detector.py` | Two-tier detector and live demo |
| `benchmark.py` | Accuracy / speed / energy ablation, writes `benchmark_results.csv` and `accuracy_vs_energy.png` |
| `report/` | Coursework report (Word) and architecture diagram |

## Limitations and future work

- The test set is small. Next steps are a larger evaluation including low-light, crowded and distant-person scenes.
- Energy figures are TDP-based estimates on machines without power counters.
- Planned: INT8 quantisation (OpenVINO/ONNX), knowledge distillation, escalation thresholds learned from data, and Raspberry Pi tests with a hardware power meter.
