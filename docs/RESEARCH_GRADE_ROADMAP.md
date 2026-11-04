# Research-Grade Roadmap

This document separates the current working prototype from extensions that would be required for a stronger dissertation or PhD-level research contribution.

## Current system

- OpenCV reads webcam, video, image and image-folder sources.
- YOLO26n detects COCO person class 0.
- Adaptive routing can invoke YOLO26s based on confidence-band, crowd, scene-statistics and periodic-audit triggers.
- YOLO26s detections replace nano detections for escalated frames.
- OpenCV draws per-frame detections and a small performance overlay.
- CodeCarbon records run-level energy and estimated emissions.
- The command-line program appends a summary to run_log.csv.
- benchmark.py compares three modes on labelled images and computes a lightweight AP approximation.

## Work package 1 — Evaluation validity

1. Integrate official Ultralytics validation and/or pycocotools.
2. Add per-class precision, recall, F1, confusion/error analysis and standard AP metrics.
3. Define video-level train/validation/test splits to reduce temporal leakage.
4. Save raw per-image predictions and evaluation metadata.
5. Add tests for IoU, matching, empty labels, class filtering and metric edge cases.

**Acceptance evidence:** evaluator outputs agree with a trusted reference implementation on a small known test fixture.

## Work package 2 — Real-time systems engineering

1. Separate capture, inference and rendering using bounded queues.
2. Measure capture-to-display latency, queue delay, dropped frames and sustained FPS.
3. Add configurable frame sampling and back-pressure.
4. Record device utilisation, memory and thermal behaviour where available.
5. Add clean shutdown, camera reconnect and structured error logging.

**Acceptance evidence:** sustained tests report p50/p95/p99 end-to-end latency and dropped-frame rate under a documented workload.

## Work package 3 — Temporal tracking

1. Add ByteTrack or BoT-SORT using a compatible Ultralytics tracking interface.
2. Measure identity switches, IDF1/HOTA or an appropriate tracking metric.
3. Distinguish per-frame detections from unique track counts.
4. Evaluate occlusion, entry/exit and crowded scenes.

**Acceptance evidence:** tracking metrics on a labelled sequence dataset. Tracking is not currently implemented.

## Work package 4 — Principled adaptive routing

Replace fixed heuristics with a learned or calibrated policy using signals such as:
- Detection score distributions and box geometry.
- Model disagreement on a labelled calibration subset.
- Temporal track instability.
- Scene embeddings or quality indicators.
- Remaining energy/latency budget.

Candidate approaches include logistic routing, gradient-boosted routing, calibrated confidence, conformal risk control or a contextual bandit. Compare all methods against the current interpretable heuristic and matched-budget baselines.

**Acceptance evidence:** held-out results showing the quality-energy-latency trade-off, calibration/risk analysis and ablations. Avoid training and evaluating the router on the same frames.

## Work package 5 — Energy-aware optimisation

Investigate:
- Dynamic input resolution.
- Frame skipping and event-triggered sampling.
- Quantisation and hardware-appropriate acceleration.
- Knowledge distillation.
- Model caching and batching where latency permits.
- Explicit energy budgets and constrained optimisation.

Use a multi-objective formulation, for example minimising energy subject to recall and latency constraints, and report the Pareto frontier. State assumptions and measurement uncertainty.

## Work package 6 — Robustness, fairness and deployment

1. Evaluate cross-camera and cross-location domain shift.
2. Report performance by scene conditions and relevant subgroups where lawful and appropriately annotated.
3. Perform privacy, security and data-retention threat modelling.
4. Test on a named edge device with sustained power/thermal measurements.
5. Document camera permissions, consent and applicable data-protection obligations.

## Suggested research questions

- How much energy can adaptive inference save at a fixed person-recall constraint?
- Which routing signals offer the best marginal quality improvement per joule?
- Does temporal context reduce unnecessary escalation under occlusion and motion blur?
- How robust is the routing policy when transferred to unseen cameras?
- How do frame rate, resolution and model tier jointly shape the quality-latency-energy Pareto frontier?

## Publication-quality evidence checklist

- [ ] Held-out video-level test set and documented split.
- [ ] Official detection evaluation.
- [ ] Strong fixed-model and matched-budget baselines.
- [ ] Ablation and threshold-sensitivity studies.
- [ ] Multiple independent runs and confidence intervals.
- [ ] p50/p95/p99 end-to-end latency.
- [ ] Energy methodology, telemetry and uncertainty documented.
- [ ] Robustness and failure-case analysis.
- [ ] Reproducible code, configurations and raw result files.
- [ ] Claims restricted to what the experiments actually establish.
