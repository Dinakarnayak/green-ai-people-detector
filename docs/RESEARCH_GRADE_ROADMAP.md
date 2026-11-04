# Research and Development Roadmap

This document separates the current laptop-based coursework prototype from optional future research. Proposed items are not implemented or validated unless supported by code and evidence.

## Current prototype
- OpenCV reads webcam, video, image and image-folder sources locally.
- YOLO26n detects COCO person class 0.
- Adaptive routing can invoke YOLO26s based on confidence-band, candidate, optional scene and periodic-audit triggers.
- OpenCV draws detections and a performance overlay.
- CodeCarbon records run-level estimated energy and emissions.
- The command-line program appends summaries to `run_log.csv`.
- `benchmark.py` compares configurations using an exploratory person-class AP approximation.

## Coursework priority — one laptop
1. Record the actual laptop CPU, GPU, RAM, operating system and power mode.
2. Run YOLO26n-only, YOLO26s-only and adaptive configurations on identical local test data.
3. Repeat runs and preserve raw CSV/log outputs.
4. Report inference latency, end-to-end FPS and CodeCarbon estimates separately.
5. Capture genuine laptop screenshots or recordings.
6. Explain AP approximation and energy-estimation limitations.
7. Keep all coursework evidence traceable to the same laptop.

## Future research extensions

### Evaluation validity
- Integrate official Ultralytics validation and/or pycocotools.
- Add error analysis and tests for matching and metric edge cases.
- Use held-out sequences and avoid temporal leakage.

### Real-time systems engineering
- Separate capture, inference and rendering using bounded queues.
- Measure capture-to-display latency, queue delay, dropped frames and sustained FPS.
- Add configurable frame sampling and back-pressure.

### Temporal tracking
- Investigate ByteTrack or BoT-SORT.
- Distinguish per-frame detections from unique track counts.
- Evaluate tracking metrics on labelled sequences.

### Adaptive routing
- Calibrate confidence and model disagreement using a separate validation set.
- Compare routing methods against matched-escalation-budget baselines.
- Perform threshold sensitivity and ablation studies.

### Energy-aware optimisation
- Investigate dynamic input resolution, frame skipping, quantisation and model distillation.
- Use a multi-objective formulation and report quality-latency-energy trade-offs.
- Any future portability study must be clearly separated from the current laptop-only coursework scope.

## Research questions
- How much estimated energy can adaptive inference save at a fixed person-recall constraint on the same laptop?
- Which routing signals offer the best marginal quality improvement per joule?
- Does temporal context reduce unnecessary escalation under occlusion and motion blur?
- How do frame rate, resolution and model tier jointly shape the quality-latency-energy trade-off?

## Evidence standard
- [ ] Same-laptop baselines and adaptive runs.
- [ ] Held-out test data.
- [ ] Standard detection evaluation for formal AP claims.
- [ ] Ablation and threshold-sensitivity studies.
- [ ] Multiple independent runs and uncertainty reporting.
- [ ] End-to-end latency percentiles if implemented and measured.
- [ ] Energy methodology and estimation limits documented.
- [ ] Reproducible code, configurations and raw results.
- [ ] Claims restricted to demonstrated results.
