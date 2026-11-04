# Laptop-only Evaluation Protocol

## Scope
All coursework demonstrations and measurements must be produced on the student's own Windows laptop. Do not use another computer, cloud VM, remote GPU or external edge device as assessed evidence.

## Record before testing
Record the actual laptop make/model, CPU, GPU (if used), RAM, Windows version, power mode, Python/PyTorch/Ultralytics/OpenCV/CodeCarbon versions, model weights, dataset version and code commit. Do not guess specifications.

## Run procedure
1. Use the same laptop and power state for every configuration.
2. Download weights and dataset before timing.
3. Warm up each model.
4. Run YOLO26n-only, YOLO26s-only and adaptive mode on identical images.
5. Perform at least five independent runs where practical.
6. Preserve all outputs; do not report only the best run.
7. Capture genuine screenshots or recordings from the laptop.

Suggested command:

```powershell
python benchmark.py --repeats 5
```

## Metrics
- Separate inference latency from end-to-end FPS.
- Report escalation rate and counts by trigger.
- The current AP implementation is exploratory, not official COCO AP.
- CodeCarbon energy and CO2e are estimates unless validated with a physical power meter.
- Energy per frame is run-level energy divided by processed frames, not a direct marginal measurement.

## Evidence checklist
- [ ] Actual laptop hardware and software metadata.
- [ ] YOLO26n-only, YOLO26s-only and adaptive runs.
- [ ] Repeated benchmark outputs and raw CSV/logs.
- [ ] Chart generated from the laptop's own benchmark.
- [ ] Genuine laptop demo screenshots/recording.
- [ ] Limitations and estimation method explained.
- [ ] No other-machine results presented as laptop evidence.

## Historical results
Earlier 128-image figures came from a 2-vCPU CPU environment. They are historical development data only and are not coursework evidence. Regenerate results on the laptop before including numerical findings in the report.
