# Experimental Protocol — Green AI People Detector

## Purpose
Compare YOLO26n, YOLO26s and adaptive routing under identical conditions.

## Controls
- Same computer, power mode and thermal conditions
- Same validation images and image order
- Same image size and confidence threshold
- Same Python and package versions
- Same camera/video resolution for live tests

## Procedure
1. Record OS, CPU, GPU, RAM, Python and package versions.
2. Run each model once to download weights and warm up inference.
3. Run each benchmark at least five times; alternate model order where practical.
4. Preserve each raw output and CSV rather than overwriting the only copy.
5. Report mean and standard deviation for latency, energy and emissions.
6. Use labelled validation data and explain the AP calculation.
7. Report adaptive escalation rate and reasons for escalation.
8. Include failure examples: missed people, false positives, occlusion, small/distant people, blur and difficult lighting.

## Interpretation cautions
- CodeCarbon provides estimated energy/emissions, not a laboratory-grade per-frame power trace.
- Energy per image is run-level energy divided by image count and includes measurement/software overhead.
- The lightweight AP implementation in benchmark.py is an approximation. For formal COCO mAP, use the official Ultralytics validation pipeline or pycocotools.
- Confidence is not accuracy.
- Lower energy is not automatically better if detection quality falls unacceptably.
- Periodic audits may recover some misses by invoking the larger model, but cannot guarantee recovery.
