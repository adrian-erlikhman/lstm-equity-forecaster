# LSTM Equity Forecaster

A **TensorFlow/Keras LSTM** that forecasts short-term price movement from
rolling-window sequences, **benchmarked against a linear-regression baseline**
so the value added by the recurrent model is measurable, not assumed.

## What it does
- Builds supervised rolling-window sequences from a price series
- Scales, trains a 2-layer LSTM, and forecasts the next step
- Reports test MAE for the LSTM **and** a linear baseline, in both scaled and
  dollar terms

## Run it
```bash
pip install -r requirements.txt
python forecast.py                 # synthetic series, no network needed
python forecast.py --csv prices.csv --look 30 --epochs 20
```
`prices.csv` just needs a `close` column (e.g. exported from `yfinance`).

## Notes
Defaults to a synthetic trend+seasonality+noise series so results are
reproducible without market-data access. Swapping in a real `close` column is a
one-flag change — the windowing, scaling, and benchmarking stay the same.

_Author: Adrian Erlikhman_
