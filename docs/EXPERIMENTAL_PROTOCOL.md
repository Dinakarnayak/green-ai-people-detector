# Experimental Protocol — Laptop-only Green AI People Detector

## 1. Scope
Compare YOLO26n-only, YOLO26s-only and adaptive inference on the student's own Windows laptop. All assessed demonstrations, benchmarks, screenshots and measurements must come from that laptop. Do not use another computer, cloud VM, remote GPU or external edge device as coursework evidence.

## 2. Objective and hypotheses
Evaluate detection quality, runtime and CodeCarbon-estimated environmental impact together.
- H1: Adaptive inference may use less estimated energy per frame than YOLO26s-only while retaining useful person recall.
- H2: Adaptive inference may use more energy than YOLO26n-only but improve detection quality on difficult frames.
- H3: Scene checks and periodic audits may identify difficult frames missed by confidence-only routing.
- H4: Routing thresholds change the quality-energy-latency trade-off.

These are hypotheses, not guaranteed findings.

## 3. Laptop and software record
Record actual values before testing; do not guess or copy another machine's specifications:
- Laptop make/model, CPU, GPU (if used), RAM and Windows version.
- Power mode and whether plugged in or on battery.
- Python, PyTorch, CUDA (if applicable), Ultralytics, OpenCV and CodeCarbon versions.
- Model weight filenames, input size, confidence threshold and dataset version.
- Code commit and whether inference uses CPU or GPU.

Use the same laptop, dataset, software environment and settings for comparable runs.

## 4. Baselines
1. YOLO26n on every image.
2. YOLO26s on every image.
3. Current adaptive policy.
4. Confidence-band-only ablation.
5. Candidate-trigger ablation.
6. Scene-trigger ablation when relevant test images are available.

Record actual escalation rates and use identical inputs.

## 5. Run procedure
1. Close unnecessary applications and keep power state consistent.
2. Download model weights and data before timed runs.
3. Warm up each model before measured runs.
4. Run all configurations on the same image set.
5. Perform at least five independent runs where practical.
6. Alternate run order where practical to reduce thermal/order effects.
7. Preserve all raw outputs; do not select only the best run.
8. Do not mix results from other machines with laptop results.

Suggested command: `python benchmark.py --repeats 5`.

## 6. Metrics
### Detection quality
The current benchmark reports exploratory person-class AP50 and AP50:95 approximations, plus precision and recall. These are not official COCO metrics. State this limitation. Use a trusted evaluator for formal AP claims.

### Runtime
Report inference time per image and end-to-end time separately. Define FPS precisely. Do not describe inference-only timing as end-to-end latency.

### Energy and emissions
Report CodeCarbon energy and CO2e as estimates. State telemetry availability and whether power was measured or estimated. Energy per frame is run-level estimated energy divided by processed frames, not a direct marginal measurement. Do not claim physical wall-power measurement without a power meter.

### Routing
Report escalation percentage and counts by reason. Do not describe confidence thresholds as calibrated probabilities unless calibration has been tested.

## 7. Evidence
- Actual laptop hardware/software metadata.
- YOLO26n-only, YOLO26s-only and adaptive runs.
- Repeated benchmark outputs and raw CSV/logs.
- Genuine screenshots or recording from the laptop.
- Chart generated from the laptop's own results.
- Limitations and energy-estimation method documented.
- No other-machine results presented as laptop evidence.

## 8. Interpretation
Do not claim energy savings or preserved detection quality until the laptop experiment supports those claims. Separate implemented code, laptop measurements, CodeCarbon estimates, illustrative images and future work.
