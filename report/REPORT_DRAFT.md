# CO3091/7091 Report Draft
## Green AI: Energy-Aware Adaptive People Detection Using YOLO26

**Student:** [Insert name]  
**Student ID:** [Insert ID]  
**Module:** CO3091/7091 — Computational Intelligence and Software Engineering: Problem Solving

> Draft for review. Verify the final word counts and references against Blackboard guidance. Replace the results placeholders with measurements you have personally reproduced. Do not submit this draft unchanged.

## 1. Introduction

Object detection allows computer systems to identify and locate objects in images and video, supporting applications such as pedestrian safety, robotics and intelligent surveillance. However, improving detection accuracy can require greater computational resources and energy. This project proposes an energy-aware people detector that uses a lightweight YOLO26 model by default and invokes a larger model only when the scene or initial detections indicate uncertainty. The approach treats energy use and detection quality as joint design objectives.

## 2. Proposed Technology

The proposed system uses Python, OpenCV, Ultralytics YOLO26 and CodeCarbon. YOLO26n is the first-stage detector and processes each input frame to locate people (COCO class 0). An adaptive routing component decides whether the result should be accepted or checked by YOLO26s, a larger second-stage model.

Escalation is triggered by several explainable signals: a person detection within a configurable confidence band, a crowded frame, difficult image conditions such as low brightness, overexposure, low contrast or blur, and a periodic audit interval. The periodic audit is important because confidence-only routing cannot escalate when the first model misses a person completely. When a trigger occurs, YOLO26s reprocesses the frame; otherwise, the YOLO26n result is retained. Bounding boxes, confidence values, selected model and escalation reasons are displayed.

CodeCarbon records run-level energy consumption and estimated carbon emissions. The benchmark compares YOLO26n-only, YOLO26s-only and adaptive modes using the same labelled images. It records person-class AP estimates, mean detector latency, wall-clock throughput, energy per image, estimated CO2 and escalation rate. The lightweight AP calculation is an approximation; formal COCO evaluation should use the official Ultralytics validation pipeline or pycocotools. Repeated runs and consistent hardware conditions are required because energy estimates depend on device telemetry and measurement duration.

## 3. Conclusion

The proposed adaptive detector aims to reduce unnecessary use of a larger model while retaining useful people-detection performance. Its main challenge is balancing energy savings against missed detections, especially for small, occluded or poorly illuminated people. Future work should validate routing thresholds on a larger dataset, compare repeated energy measurements, and investigate quantisation or distillation on supported edge hardware.

## References

[1] Schwartz, R. et al. (2020) 'Green AI', *Communications of the ACM*, 63(12), pp. 54–63. https://doi.org/10.1145/3381831

[2] Lin, T.-Y. et al. (2014) 'Microsoft COCO: Common Objects in Context', *Computer Vision – ECCV 2014*, pp. 740–755. https://doi.org/10.1007/978-3-319-10602-1_48

[3] Ultralytics (n.d.) *YOLO26 Documentation*. Available at: https://docs.ultralytics.com/models/yolo26/ (Accessed: [insert date accessed]).

[4] CodeCarbon (n.d.) *CodeCarbon Documentation*. Available at: https://docs.codecarbon.io/ (Accessed: [insert date accessed]).

## Experimental results to insert after running

| Mode | Person AP50 | Mean latency (ms) | Energy/image (J) | CO2/1,000 images (g) |
|---|---:|---:|---:|---:|
| YOLO26s only | [measured] | [measured] | [measured] | [measured] |
| YOLO26n only | [measured] | [measured] | [measured] | [measured] |
| Adaptive | [measured] | [measured] | [measured] | [measured] |

Record the computer hardware, operating system, Python/package versions, dataset split, number of images, image size, confidence threshold, repeat count and summary statistics. The repository's existing CSV is not automatically evidence of a reproducible experiment; rerun the benchmark and preserve your own raw outputs before citing values.
