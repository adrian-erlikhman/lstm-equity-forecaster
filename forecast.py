"""
LSTM Equity Forecaster
======================
A TensorFlow/Keras LSTM that forecasts short-term price movement from
rolling-window sequences, benchmarked against a linear-regression baseline.

Runs on a synthetic price series (trend + seasonality + noise) by default so it
works with no network access. Use real data with:
    python forecast.py --csv prices.csv     # needs a 'close' column
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def synth_series(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.02 * t
    season = 5 * np.sin(2 * np.pi * t / 50) + 2 * np.sin(2 * np.pi * t / 13)
    noise = rng.normal(0, 1.2, n).cumsum() * 0.3
    return 100 + trend + season + noise


def make_windows(series, look):
    X, y = [], []
    for i in range(len(series) - look):
        X.append(series[i:i + look])
        y.append(series[i + look])
    return np.array(X), np.array(y)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", help="CSV with a 'close' column")
    p.add_argument("--look", type=int, default=30, help="look-back window")
    p.add_argument("--epochs", type=int, default=15)
    args = p.parse_args()

    series = (pd.read_csv(args.csv)["close"].values.astype(float)
              if args.csv else synth_series())

    scaler = MinMaxScaler()
    s = scaler.fit_transform(series.reshape(-1, 1)).ravel()
    X, y = make_windows(s, args.look)
    split = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    # --- linear baseline ---
    lin = LinearRegression().fit(Xtr, ytr)
    lin_pred = lin.predict(Xte)

    # --- LSTM ---
    import tensorflow as tf  # imported here so the baseline works even without TF
    from tensorflow.keras import layers, models
    tf.random.set_seed(42)

    model = models.Sequential([
        layers.Input((args.look, 1)),
        layers.LSTM(48, return_sequences=True),
        layers.LSTM(24),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(Xtr[..., None], ytr, validation_split=0.1,
              epochs=args.epochs, batch_size=32, verbose=2)
    lstm_pred = model.predict(Xte[..., None], verbose=0).ravel()

    inv = lambda a: scaler.inverse_transform(a.reshape(-1, 1)).ravel()
    print("\n--- Test MAE (lower is better) ---")
    print(f"scaled : LSTM={mean_absolute_error(yte, lstm_pred):.4f}  "
          f"Linear={mean_absolute_error(yte, lin_pred):.4f}")
    print(f"price $: LSTM={mean_absolute_error(inv(yte), inv(lstm_pred)):.3f}  "
          f"Linear={mean_absolute_error(inv(yte), inv(lin_pred)):.3f}")


if __name__ == "__main__":
    main()
