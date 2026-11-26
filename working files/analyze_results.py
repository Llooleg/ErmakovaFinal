import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/oleg/PycharmProjects/Ermakov/Ermakova10SEM/finallab/ErmakovaFinal")
data = pd.read_csv(ROOT / "total_incidents.csv")
y = data["Кол. инцидентов"].to_numpy(dtype=float)
x = np.arange(1, len(y) + 1, dtype=float)
train = y[:60]
mu = train.mean()

GA_BIAS = -0.6037238731061447
GA_GAMMAS = np.array([-0.1294236048362728, -0.10230646344638802, 0.45])
feature = (1 + np.arange(1, 61) / 15.0) ** (-3)
a, b = np.linalg.lstsq(np.column_stack([np.ones(60), feature]), train, rcond=None)[0]
cap = max(35.0, float(train.max() + 4 * np.std(train)))

def predict_ga(t, history):
    previous = np.asarray(history[-3:], dtype=float)
    if len(previous) < 3:
        previous = np.r_[np.full(3 - len(previous), mu), previous]
    inverse = a + b / (1 + t / 15.0) ** 3
    dynamic = np.dot(GA_GAMMAS, previous[::-1] - mu)
    return float(np.clip(inverse + GA_BIAS + dynamic, 0, cap))

forecast_full = []
history = []
for t in range(1, 105):
    estimate = predict_ga(t, history)
    forecast_full.append(estimate)
    history.append(float(y[t - 1]) if t <= 101 else estimate)
forecast_full = np.asarray(forecast_full)
ga = forecast_full[:101]

training_eval = np.arange(3, 59)
lookups = np.arange(1, 105, dtype=float)
shifts = np.round(np.arange(0, 1.001, 0.05), 8)
scores = [np.mean(np.abs(y[training_eval] - np.interp(x[training_eval] + delta, lookups, forecast_full))) for delta in shifts]
shift = float(shifts[np.argmin(scores)])
shifted = np.interp(x + shift, lookups, forecast_full)

linear_coef = np.polyfit(x[:60], train, 1)
linear = np.polyval(linear_coef, x)
mean_forecast = np.full(len(y), mu)
naive = np.r_[np.nan, y[:-1]]

def metrics(truth, pred):
    error = truth - pred
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(np.mean(error ** 2)),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_pct": float(100 * np.mean(np.abs(error / truth))),
        "error_variance_ddof1": float(np.var(error, ddof=1)),
        "within_5pct_of_actual_pct": float(100 * np.mean(np.abs(error / truth) <= 0.05)),
        "within_5pct_of_prediction_pct": float(100 * np.mean(np.abs(error) <= 0.05 * np.abs(pred))),
        "bias_mean_error": float(np.mean(error)),
    }

models = {"mean_train": mean_forecast, "linear": linear, "ga_one_step": ga, "ga_shifted_retrospective": shifted, "naive_previous": naive}
result = {
    "data": {
        "rows": len(y),
        "duplicates_timestamps": int(data["Дата"].duplicated().sum()),
        "missing_values": data.isna().sum().to_dict(),
        "mean_all": float(y.mean()),
        "mean_train": float(mu),
        "mean_test": float(y[60:].mean()),
        "std_all_ddof1": float(y.std(ddof=1)),
        "min": float(y.min()),
        "max": float(y.max()),
        "train_autocorrelation_lags_1_2_3": [float(np.corrcoef(train[k:], train[:-k])[0, 1]) for k in (1, 2, 3)],
    },
    "inverse_coefficients": {"a": float(a), "b": float(b), "cap": cap},
    "shift": {"steps_left": shift, "seconds_left": 30 * shift, "train_mae_no_shift": scores[0], "train_mae_shifted": min(scores)},
    "test_61_101": {name: metrics(y[60:], pred[60:]) for name, pred in models.items()},
    "train_4_59": {name: metrics(y[training_eval], pred[training_eval]) for name, pred in models.items() if name != "naive_previous"},
}
output = ROOT / "working files/recomputed_metrics.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
