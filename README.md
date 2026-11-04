# Green AI People Detector

An energy-aware people-detection system built with **YOLO26** and **CodeCarbon**. The system uses a lightweight-first, two-tier pipeline: YOLO26n processes every frame and YOLO26s is invoked only when configured escalation rules indicate that a frame may need a closer look.

**Evaluation scope: one Windows laptop only.** All demonstrations, benchmarks, screenshots and performance measurements submitted for this project must be generated on the student's own laptop. No Raspberry Pi, cloud VM, remote GPU, second computer or external edge device is part of the evaluated implementation.

The project investigates:

> Can adaptive model selection reduce estimated inference energy while retaining useful person-detection performance on a single laptop?

![System architecture](report/system_diagram.png)

## Demo preview

![Green AI YOLO26 real-time people detection dashboard](docs/images/realtime_sample_readme.jpg)

*Illustrative dashboard prototype only. FPS, people count, energy and CO₂ values shown are sample values, not measurements from a validated live run. Do not submit this image as experimental evidence.*

## Key features

- **Adaptive two-tier inference:** YOLO26n is the default model; YOLO26s handles selected escalated frames.
- **Confidence-based escalation:** uncertain detections and weak person candidates can trigger the larger model.
- **Optional scene checks:** dark or blurred frames can be escalated with `--scene`.
- **Periodic safety check:** video frames can be escalated at a configurable interval.
- **Energy tracking:** CodeCarbon records run-level energy and estimated emissions.
- **Run logging:** summaries are printed and appended to `run_log.csv`.
- **Benchmarking:** compare model configurations on the same labelled image set on the same laptop.

## How it works

1. A frame enters Tier 1, where YOLO26n performs person detection.
2. The detector checks confidence values and configured escalation conditions.
3. If a trigger fires, Tier 2 (YOLO26s) re-processes that frame.
4. The system draws detections and records timing, escalation and energy information.

### Escalation rules

| Trigger | Condition | Default |
|---|---|---|
| `uncertain` | Person confidence is between 0.25 and 0.50 | Enabled |
| `candidate` | Weak person candidate is between 0.10 and 0.25 | Enabled |
| `scene` | Frame is very dark (mean brightness below 40) or blurred (Laplacian variance below 30) | Disabled; enable with `--scene` |
| `periodic` | Every Nth frame is sent to the larger model as a safety check | Every 30 frames; disable with `--every 0` |

Thresholds can be adjusted from the command line. Green boxes indicate detections handled by YOLO26n; orange boxes indicate an escalated frame, with the trigger reason shown on screen.

## Experimental results status

**Status: laptop validation pending.**

The committed `benchmark_results.csv` currently contains a 128-image exploratory benchmark collected in a 2-vCPU CPU environment. It is retained only as historical development data and is **not valid evidence for the laptop-only coursework evaluation**. The file is stale until you regenerate it on your own laptop; do not cite its values as your laptop results.

Run the benchmark on the same Windows laptop used for the demonstration:

```powershell
python benchmark.py --repeats 5
```

Until that run is completed, this README makes no claim about the laptop's measured accuracy, FPS, latency, energy use or CO₂e. Use the generated laptop results and raw logs when completing the report. The benchmark's AP calculation is a lightweight approximation, not official Ultralytics or pycocotools COCO evaluation; label it accordingly.

CodeCarbon provides estimated energy and emissions. Unless a physical power meter is used, do not describe these estimates as direct wall-power measurements. State the laptop hardware, software versions, power mode, measurement method and run conditions.

## Requirements

- Python 3.9 or later recommended
- Windows laptop for the assessed demonstration (the code may also run on other operating systems, but those are outside the assessed setup)
- Webcam, video file or image folder
- Internet access on first run to download model weights and the default COCO128 dataset

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

YOLO26 weights (`yolo26n.pt` and `yolo26s.pt`) are downloaded automatically by Ultralytics when first required.

## Run the detector

```powershell
# Adaptive mode with the default webcam
python green_detector.py

# Lightweight model only
python green_detector.py --mode nano

# Larger baseline model only
python green_detector.py --mode small

# Enable dark/blurred-scene escalation
python green_detector.py --scene

# Process a local video and escalate every 15 frames
python green_detector.py --source video.mp4 --every 15

# Process an image folder without opening a display window
python green_detector.py --source images/ --no-show --save outputs/demo

# Run a bounded local webcam demo
python green_detector.py --max-frames 300 --save outputs/laptop-demo
```

Press **Q** to close a live display. Each run prints a summary and appends it to `run_log.csv`.

## Reproduce the laptop benchmark

```powershell
python benchmark.py --repeats 5
```

Use the same laptop, dataset, image size, model weights and software environment for all configurations. Warm up models, keep background load and power mode consistent, and report all independent runs rather than selecting the best one. See [Laptop-only evaluation protocol](docs/LAPTOP_ONLY_EVALUATION.md).

The benchmark writes:
- `benchmark_results.csv` — configuration-level results from the machine on which it was run.
- `accuracy_vs_energy.png` — accuracy-versus-estimated-energy chart.

Do not overwrite or mix results from different machines without clearly separating them.

## Suggested demonstration

1. Run YOLO26s-only and show the baseline detections.
2. Run YOLO26n-only and compare detection quality and speed.
3. Run adaptive mode and show when escalation occurs.
4. Use a local dark or blurred video/image only if demonstrating the optional scene trigger.
5. Show the actual laptop-generated `run_log.csv`, benchmark CSV and terminal output.
6. Explain the difference between inference latency, end-to-end FPS and CodeCarbon energy estimates.

## Repository structure

| Path | Description |
|---|---|
| `green_detector.py` | Adaptive two-tier detector and live demo |
| `benchmark.py` | Accuracy, speed and estimated-energy comparison |
| `requirements.txt` | Python dependencies |
| `benchmark_results.csv` | Currently contains historical 2-vCPU results; do not treat as laptop evidence. Regenerate locally before reporting |
| `docs/EXPERIMENTAL_PROTOCOL.md` | Experimental design and reporting guidance |
| `docs/LAPTOP_ONLY_EVALUATION.md` | Required laptop-only evidence and run procedure |
| `docs/RESEARCH_GRADE_ROADMAP.md` | Clearly separated future work |
| `docs/images/realtime_sample_readme.jpg` | Illustrative dashboard only |
| `report/system_diagram.png` | System architecture diagram |
| `report/` | Coursework report materials |

## Limitations

- COCO128 is a small evaluation set and does not represent all real-world people-detection conditions.
- The repository's AP implementation is an exploratory approximation; formal reporting requires a validated evaluator such as Ultralytics validation and/or pycocotools.
- CodeCarbon energy and CO₂e are estimates and depend on the hardware telemetry and estimation method available.
- Further evaluation should include crowded scenes, occlusion, low light, motion blur and small/distant people, using local data on the same laptop.
- External edge-device deployment is outside this coursework's demonstration scope and must not be described as completed.

## Research integrity

Separate implemented features, measured laptop results, estimated values, illustrative material and proposed future work. Do not describe a prototype image as a live screenshot. Record the code commit, model weights, dataset version, laptop specifications, software environment and command used for every experiment.
