# Running OOD/Anomaly Detection Experiments

## Setup

```bash
conda activate timeseries
cd /path/to/EasyTSAD
```

## Run an experiment

```bash
python Examples/run_your_algo/runKANAD.py
```

Each script trains the model, evaluates it, and saves plots.

## Methods available

| Method | Script |
|--------|--------|
| KANAD  | `runKANAD.py` |
| CATCH  | `runCATCH.py` |
| PGRF   | `runPGRF.py` |
| CAD    | `runCAD.py` |

Other methods (AE, TranAD, TimesNet, etc.) live in `EasyTSAD/Methods/` — add a run script following the pattern below.

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
