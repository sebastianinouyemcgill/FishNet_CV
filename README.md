# FishNet CV — Fish Length Estimation

Research-oriented Python scaffold for estimating **physical fish length (mm)** from RGB images and **YOLO polygon** annotations (fish, blue calibration rectangles, yellow calibration rectangles). Geometry and calibration follow the course assignment; see **`assignment.pdf`** (local only, gitignored) for full requirements.

## Goals

| Input | Output |
|--------|--------|
| RGB fish images | `predictions.csv`: `image_id`, `predicted_length_mm` |
| YOLO seg polygons: fish, blue/yellow 100 mm markers | Optional figures, metrics (MAE, RMSE) |

**Class IDs** (from `data/fishnet.yaml` when using the FishNet archive):

| ID | Name |
|----|------|
| 0 | blue calibration rectangle |
| 1 | yellow calibration rectangle |
| 2 | fish |

Calibration rectangles are assumed to be **100 mm** long (`CALIBRATION_RECT_MM` in `src/config.py`).

---

## Requirements

- **Python 3.11+**
- See `requirements.txt` or install with `pip install -e .`

Core libraries: `numpy`, `opencv-python`, `matplotlib`, `pandas`, `scikit-image`, `scikit-learn`, `scipy`, `tqdm`, `PyYAML`, `jupyter`.

Optional (commented in `requirements.txt`): `ultralytics`, `torch`, `transformers`.

---

## Environment setup

```bash
cd fishnet_cv
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# Or editable install:
pip install -e ".[dev]"
```

Register the Jupyter kernel (optional):

```bash
python -m ipykernel install --user --name fishnet-cv --display-name "FishNet CV"
```

Run tests:

```bash
pytest tests/ -v
```

---

## Dataset placement

The repo may contain data locally, but **`data/` and `assignment.pdf` are gitignored**.

### Recommended layout

```
data/
├── raw/
│   └── fishnet/              # extracted dataset
│       ├── images/
│       │   ├── train/
│       │   ├── valid/        # or val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── valid/
│           └── test/
├── processed/                # cached masks, rectified images
└── annotations/              # ground-truth CSV for evaluation
    └── lengths_mm.csv        # columns: image_id, length_mm
```

### If you already have `data/fishnet/`

`src/config.py` auto-detects `data/fishnet` when `data/raw/fishnet` is absent (current clone layout).

Extract the archive if needed:

```bash
tar -xzf data/fishnet_dataset.tar.gz -C data/raw/
# Ensure images/ and labels/ exist under data/raw/fishnet/
```

Label format: one line per instance — `class_id x1 y1 x2 y2 ...` with **normalized** polygon coordinates (YOLO segmentation).

---

## Example commands

### Legacy mode (unchanged)

Writes to `outputs/predictions/predictions.csv` — existing notebooks still work:

```bash
python main.py --split test --method pca
python main.py --split test --method skeleton --limit 5 --visualize
```

### Managed experiment runs (recommended)

Each run gets an isolated directory under `outputs/runs/<run_name>/`:

```bash
python main.py --pipeline baseline --method bbox --run-name baseline_bbox_v1 --split valid
python main.py --pipeline advanced --method pca --perspective --run-name advanced_pca_rect_v1 --split valid
python main.py --pipeline baseline --method pca --run-name baseline_pca_v1 --overwrite  # replace existing
```

Auto-generated run name when omitted:

```bash
python main.py --pipeline baseline --method pca --split valid
# → outputs/runs/baseline_pca_2026-05-30_143500/
```

### Notebook-first API (primary workflow)

```python
from src.experiments import run_experiment, run_experiments

run = run_experiment(
    pipeline="baseline",
    method="bbox",
    split="valid",
    run_name="baseline_bbox_v1",
)

results_df = run_experiments([
    {"pipeline": "baseline", "method": "bbox", "run_name": "baseline_bbox_v1", "split": "valid"},
    {"pipeline": "baseline", "method": "pca", "run_name": "baseline_pca_v1", "split": "valid"},
    {"pipeline": "advanced", "method": "pca", "perspective": True, "run_name": "advanced_pca_rect_v1"},
])
```

See `notebooks/03_run_experiments.ipynb` and `notebooks/04_analyze_results.ipynb`.

Environment overrides (legacy mode):

```bash
export FISHNET_SPLIT=test
export FISHNET_METHOD=pca
export FISHNET_PERSPECTIVE=1
python main.py
```

**Evaluation** (legacy path — still supported):

```python
from pathlib import Path
from src.evaluation import evaluate_run

evaluate_run(
    ground_truth_path=Path("data/annotations/validation_lengths.csv"),
    predictions_path=Path("outputs/predictions/predictions.csv"),
    method="pca",
    split="valid",
)
```

**Evaluation** (run directory):

```python
evaluate_run(
    ground_truth_path=Path("data/annotations/validation_lengths.csv"),
    predictions_path=Path("outputs/runs/baseline_bbox_v1/predictions.csv"),
    output_dir=Path("outputs/runs/baseline_bbox_v1"),
)
```

---

## Repository architecture

```
fishnet_cv/
├── data/                 # gitignored — raw, processed, annotations
├── notebooks/            # exploratory + experiment workflows
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── masks.py
│   ├── calibration.py
│   ├── measurement.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── utils.py
│   ├── pipelines/        # baseline vs advanced runners
│   │   ├── base.py       # shared inference loop
│   │   ├── baseline.py
│   │   ├── advanced.py
│   │   └── registry.py
│   └── experiments/      # RunManager + notebook API
│       ├── run_manager.py
│       ├── results.py
│       └── __init__.py   # run_experiment, run_experiments
├── outputs/
│   ├── runs/             # isolated experiment runs
│   │   ├── experiments.csv   # master registry (append-only)
│   │   └── <run_name>/
│   ├── predictions/      # legacy flat output
│   ├── figures/
│   └── metrics/
├── tests/
├── main.py               # CLI (legacy + managed runs)
├── requirements.txt
└── pyproject.toml
```

### Module responsibilities

| Module | Role |
|--------|------|
| `dataset.py` | Recursive image/label discovery, `DatasetSample`, splits |
| `masks.py` | Rasterization, morphology, skeleton, overlays |
| `calibration.py` | Marker scale (px/mm), optional `findHomography` + warp |
| `measurement.py` | Pluggable length methods; `register_measurement_method()` |
| `evaluation.py` | CSV merge, MAE/RMSE; optional `output_dir` per run |
| `visualization.py` | Overlays, PCA axis, homography plots |
| `pipelines/` | `BaselinePipeline` (geometry only) vs `AdvancedPipeline` (perspective, etc.) |
| `experiments/` | `RunManager`, `run_experiment()`, registry CSV |

---

## Experiment workflow (notebook-first)

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `dataset_exploration.ipynb` | Verify data, masks, polygons |
| 2 | `annotation_helper.ipynb` | Manual lengths → `validation_lengths.csv` |
| 3 | **`03_run_experiments.ipynb`** | Run baseline + advanced experiments |
| 4 | **`04_analyze_results.ipynb`** | Compare MAE/RMSE, per-image errors |

Legacy notebooks (`baseline_measurement.ipynb`, `perspective_correction.ipynb`, `evaluation.ipynb`) still work unchanged.

### Baseline vs advanced

| | Baseline | Advanced |
|---|----------|----------|
| **Pipeline** | `pipeline="baseline"` | `pipeline="advanced"` |
| **Methods** | bbox, pca, skeleton | same + extensions |
| **Perspective** | always off | optional `perspective=True` |
| **ML** | none | future hooks |

Both reuse `src/pipelines/base.py` inference loop.

### Run directory layout

```
outputs/runs/baseline_bbox_v1/
├── predictions.csv
├── comparison.csv      # if ground truth available
├── metrics.json
├── config.json
├── figures/            # optional (--visualize)
└── debug/
```

Master registry: `outputs/runs/experiments.csv` (append-only, never deletes).

---

## Migration summary

### Old workflow

```bash
python main.py --split valid --method bbox
# → outputs/predictions/predictions.csv (overwrites)
evaluate_run(gt, predictions)  # → outputs/metrics/comparison.csv (overwrites)
```

### New workflow

```python
from src.experiments import run_experiment
run_experiment(pipeline="baseline", method="bbox", run_name="baseline_bbox_v1", split="valid")
# → outputs/runs/baseline_bbox_v1/  (isolated, no overwrite unless overwrite=True)
# → appends row to outputs/runs/experiments.csv
```

### Files added

- `src/pipelines/` — baseline, advanced, registry, shared base
- `src/experiments/` — RunManager, results, `run_experiment` API
- `notebooks/03_run_experiments.ipynb`, `04_analyze_results.ipynb`
- `outputs/runs/.gitkeep`

### Files modified

- `main.py` — thin CLI; legacy mode when no `--pipeline`/`--run-name`
- `src/evaluation.py` — optional `output_dir`; still writes legacy paths when omitted
- `README.md` — this section

### Backwards compatibility

- `python main.py --split test --method pca` — unchanged
- `main.run_pipeline()` — preserved, delegates to shared inference
- `evaluate_run()` without `output_dir` — still writes `outputs/metrics/comparison.csv`
- `experiments.jsonl` — still appended on every evaluation
- All existing notebooks run without modification

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `dataset_exploration.ipynb` | Load samples, visualize polygons and masks |
| `annotation_helper.ipynb` | Manual validation lengths CSV |
| **`03_run_experiments.ipynb`** | Run and compare experiments (primary) |
| **`04_analyze_results.ipynb`** | Registry tables, plots, per-image errors |
| `baseline_measurement.ipynb` | Legacy exploratory baseline |
| `perspective_correction.ipynb` | Homography experiments |
| `evaluation.ipynb` | Legacy flat-path evaluation |

Open from repo root so `import src` resolves, or run:

```bash
cd fishnet_cv && jupyter lab notebooks/
```

---

## Extending measurement methods

1. Implement `def my_method(mask: np.ndarray) -> float` returning **pixels**.
2. Register: `register_measurement_method("my_method", my_method)`.
3. Add CLI choice in `main.py` and document in this README.

Use `pixels_to_mm()` with `CalibrationResult.pixels_per_mm` for metric output.

---

## Outputs

| Path | Content |
|------|---------|
| `outputs/runs/<run_name>/` | Isolated run: predictions, metrics, comparison, config |
| `outputs/runs/experiments.csv` | Master registry (append-only) |
| `outputs/predictions/predictions.csv` | Legacy flat predictions |
| `outputs/figures/` | Legacy figures (or `<run>/figures/` with `--visualize`) |
| `outputs/metrics/comparison.csv` | Legacy evaluation output |
| `outputs/metrics/experiments.jsonl` | Legacy JSONL log (still appended) |

---

## Development notes

- Starter code includes **`TODO`** markers for assignment-specific tuning (homography corners, skeleton geodesic, GT CSV schema).
- Type hints and docstrings are used throughout; logging is enabled via `src/utils.setup_logging()`.
- VS Code / Cursor: select the `.venv` interpreter and use the integrated test explorer with `pytest`.

---

## License

MIT (adjust as needed for your course submission).
