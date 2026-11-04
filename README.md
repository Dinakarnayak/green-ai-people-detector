# Green AI People Detector (YOLO26 + CodeCarbon)

An energy-aware, two-tier people detector for CO3091/7091.

- **Tier 1: YOLO26n (nano)** runs on every frame.
- **Tier 2: YOLO26s (small)** runs only when YOLO26n is uncertain, meaning it finds a person whose confidence falls inside the "uncertain band" (default 0.25–0.50).
- **CodeCarbon** measures the energy (kWh) and estimated CO₂ for the whole run.

Green boxes come from YOLO26n. Orange boxes mean the frame was escalated to YOLO26s.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

The model weights (`yolo26n.pt`, `yolo26s.pt`) download automatically the first time you run the script.

## Run

```bash
python green_detector.py                    # webcam, adaptive two-tier mode (press q to quit)
python green_detector.py --mode nano        # YOLO26n only
python green_detector.py --mode small       # YOLO26s only (baseline)
python green_detector.py --source video.mp4
python green_detector.py --source images/ --no-show --save out/
python green_detector.py --band 0.25 0.6    # escalate more often
```

Each run prints a summary with frames, escalation rate, FPS, energy per frame (J) and CO₂, and appends one row to `run_log.csv`.

## Demo plan for the TA (bonus marks)

1. Run `--mode small` on the webcam for about 30 s. This is the baseline.
2. Run `--mode nano` for about 30 s.
3. Run the default adaptive mode for about 30 s.
4. Open `run_log.csv` and compare energy per frame, FPS and escalation rate across the three runs.

## Note on energy figures

CodeCarbon reads real power counters when they are available, such as Intel RAPL on Linux or NVIDIA GPUs. Otherwise it estimates power from the CPU's TDP. Run all modes on the same machine, with the laptop plugged in and other apps closed, so the comparison is fair.
