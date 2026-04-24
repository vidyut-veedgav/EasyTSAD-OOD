# Running OOD/Anomaly Detection Experiments

## Setup

```bash
conda activate timeseries
cd /path/to/EasyTSAD
```

## Methods available

All four implemented MTS (multivariate) methods have run scripts. The remaining 24 methods in `EasyTSAD/Methods/` are univariate-only and do not have run scripts yet.

| Method | Script | Type |
|--------|--------|------|
| KANAD  | `Examples/run_your_algo/runKANAD.py` | Multivariate |
| CATCH  | `Examples/run_your_algo/runCATCH.py` | Multivariate |
| PGRF   | `Examples/run_your_algo/runPGRF.py`  | Multivariate |
| CAD    | `Examples/run_your_algo/runCAD.py`   | Multivariate |

## Run all experiments

```bash
python Examples/run_your_algo/run_all.py
```

This runs KANAD → CATCH → PGRF → CAD in sequence. Each script trains the model, evaluates it, and saves plots. A summary of any failures is printed at the end.

To run a single method:

```bash
python Examples/run_your_algo/runKANAD.py
```

## Datasets

**MTS (multivariate):** `SMD`, `MSL`, `SMAP`, `SWaT`, `PSM`  
**UTS (univariate):** `AIOPS`, `NAB`, `TODS`, `UCR`, `WSD`

## Mix and match methods / datasets

Edit the relevant fields in any run script:

```python
gctrl.set_dataset(
    dataset_type="MTS",                          # or "UTS"
    dirname="/Users/vidyutveedgav/EasyTSAD/dataset",
    datasets=["MSL", "PSM"],                     # pick any subset
)

gctrl.run_exps(
    method="KANAD",                              # string matching your class name
    training_schema="naive",
    hparams={
        "batch_size": 64,
        "window": 64,
        "order": 4,
        "epochs": 20,
        "lr": 1e-3,
    },
    preprocess="z-score",                        # or "min-max" or "raw"
)
```

## Speed tips (CPU / MacBook)

- Start with a single small dataset: `datasets=["MSL"]`
- Lower epochs for a quick sanity check: `"epochs": 3`
- Run directly (not via `conda run`) so you see live progress bars:
  ```bash
  conda activate timeseries
  python Examples/run_your_algo/runKANAD.py
  ```
