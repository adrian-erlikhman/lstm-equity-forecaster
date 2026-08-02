"""
Forecast-quality metrics for time-series price prediction.
Beyond plain error: `directional_accuracy` is what actually matters for trading —
did the model get the *direction* of the next move right?
"""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def directional_accuracy(y, p):
    """Fraction of steps where the predicted move has the same sign as the real move."""
    dy = np.sign(np.diff(np.asarray(y, float)))
    dp = np.sign(np.diff(np.asarray(p, float)))
    return float(np.mean(dy == dp))


def summary(y, p):
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE%": mape(y, p),
            "DirAcc": directional_accuracy(y, p)}
