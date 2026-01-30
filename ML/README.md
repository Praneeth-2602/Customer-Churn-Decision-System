# ML Module

This folder contains dataset-agnostic ML pipeline components.

- `configs/` — per-dataset YAML configurations (e.g., `telco.yaml`, `bank.yaml`).
- `data/` — dataset CSVs used for training and testing.
- `models/` — trained artifacts are saved here as `{dataset}_model.pkl` and `{dataset}_preprocessor.pkl`.
- `preprocess.py`, `train.py`, `predict.py`, `explain.py`, `actions.py`, `simulate.py` — core pipeline modules.

Run `python train.py <dataset>` to train, `python predict.py` to run prediction helpers.
