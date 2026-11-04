# Experimental Protocol — Green AI People Detector

## 1. Study objective

Compare YOLO26n-only, YOLO26s-only and adaptive YOLO26n-to-YOLO26s inference for person detection. Evaluate detection quality, computational performance and estimated environmental impact jointly.

## 2. Experimental hypotheses

- H1: Adaptive inference uses less energy per processed frame than YOLO26s-only under a pre-defined non-inferiority margin for person recall/AP.
- H2: Adaptive inference has higher energy use than YOLO26n-only but may improve detection quality on difficult frames.
- H3: Periodic audits and scene-aware signals recover some difficult cases that confidence-only routing cannot identify.
- H4: Routing thresholds change the quality-energy-latency trade-off and produce a Pareto frontier rather than a single universally optimal setting.

These are hypotheses to test, not expected or guaranteed findings.

## 3. Dataset and leakage control

- Identify dataset source, licence, version, annotation format and person class mapping.
- Split by source video, camera, site or recording session where possible. Avoid random frame-level splits when adjacent frames are near duplicates.
- Keep a held-out test set untouched until policy and threshold selection are complete.
- Report image/sequence counts, person-instance counts, object-size distribution and scene-condition distribution.
- Preserve annotation provenance and any excluded or corrupted samples.

## 4. Systems controls

Record:
- CPU, GPU, RAM, storage, operating system and power mode.
- Python, PyTorch, CUDA, Ultralytics, OpenCV and CodeCarbon versions.
- Model weight filenames and hashes.
- Input resolution, frame rate, image size, confidence threshold and source resolution.
- Thermal state, warm-up procedure, background load and whether the device is plugged in.

Use the same hardware, data, image size, model versions and software environment across comparable runs.

## 5. Baselines and policies

At minimum:
1. YOLO26n on every frame.
2. YOLO26s on every frame.
3. Current adaptive policy.
4. Fixed periodic escalation with a matched escalation budget.
5. Random escalation with a matched escalation budget.

For a fair comparison, match escalation budgets where possible and report the actual escalation rate. The current implementation's adaptive policy uses confidence-band, crowd, scene-statistic and periodic-audit triggers.

## 6. Ablation and sensitivity experiments

Run the full adaptive policy, then remove one component at a time:
- No confidence-band trigger.
- No crowd trigger.
- No scene-quality trigger.
- No periodic audit.
- Confidence-only policy.

Sweep:
- Confidence threshold and uncertainty-band boundaries.
- Audit interval.
- Crowd threshold.
- Image size.
- Scene brightness, contrast and blur thresholds.

Choose settings using validation data only. Freeze settings before test evaluation.

## 7. Metrics

### Detection quality
Use a standard evaluator to report person-class AP50 and AP50:95, precision, recall, F1, false-positive count and false-negative count. Describe matching rules and confidence thresholds.

### Runtime
Report mean, median, p95 and p99 inference latency; end-to-end latency; sustained throughput; warm-up time; and dropped/late frames if measured. Clearly define each timing boundary.

### Energy and emissions
Report total energy, energy per frame, estimated CO2 and CO2 per 1,000 frames. State telemetry availability, emissions factor/region, measurement interval and included hardware components. Treat energy per frame as run-level energy divided by frame count, not a direct marginal measurement.

### Routing
Report escalation rate, escalation reasons and quality metrics stratified by escalated/non-escalated frames. Do not infer that an escalation trigger is a correct uncertainty estimate without validation.

## 8. Repeats and statistical analysis

- Warm up models before measured runs.
- Perform at least five independent runs for initial comparisons; more may be needed for stable estimates.
- Alternate or randomise baseline order to reduce thermal/order effects.
- Report mean, standard deviation and confidence intervals.
- Use paired comparisons on the same sequences.
- Bootstrap at the sequence/video level where possible; adjacent frames are correlated and should not be treated as independent samples.
- Report effect sizes and predefine practical significance/non-inferiority margins.

## 9. Robustness and subgroup analysis

Stratify results by:
- Low light and overexposure.
- Blur and camera motion.
- Occlusion and crowd density.
- Small/distant people.
- Indoor/outdoor scenes and camera viewpoints.
- Dataset/site/camera domain.

Inspect representative false positives and false negatives. Evaluate whether error rates differ across relevant environments and groups where lawful, appropriate and supported by reliable annotations.

## 10. Reproducibility package

Archive:
- Exact code commit and environment lock/export.
- Dataset version and split manifest.
- Model weights or their identifiers and hashes.
- Command lines and configuration.
- Raw per-run CSV/log files.
- Evaluation outputs and analysis scripts.
- Hardware/software metadata.
- Failure-case examples and limitations.

## 11. Interpretation

Do not claim energy savings without repeated measurements and uncertainty estimates. Do not claim preserved quality without a pre-specified margin and held-out evaluation. Distinguish implemented code from proposed work. The repository's lightweight AP approximation is exploratory only; use official Ultralytics validation and/or pycocotools for formal reporting.
