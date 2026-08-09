# 🧠 LSTM Equity Forecaster

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white) ![License](https://img.shields.io/badge/license-MIT-black)

A stacked LSTM for next-step price moves — kept honest by a linear baseline and scored on **directional accuracy**, not just error.

```mermaid
flowchart LR
    A[Price series] --> B[Rolling windows<br/>look-back = 30]
    B --> C[MinMax scale]
    C --> D[2-layer LSTM<br/>48 → 24 → 1]
    C --> E[Linear baseline]
    D --> F[MAE · RMSE ·<br/>directional accuracy]
    E --> F
```

**How** — rolling windows → 2-layer LSTM, compared to linear regression on every run.
**Why it's honest** — a model that echoes yesterday's price scores a great MAE and coin-flip *direction*. Reporting directional accuracy alongside the baseline is the difference between a demo and an experiment.

## Run
```bash
pip install -r requirements.txt
python forecast.py                              # synthetic series
python forecast.py --csv prices.csv --epochs 20 # your data: 'close' column
```

<sub>MIT · Adrian Erlikhman · synthetic series by default for reproducibility</sub>
