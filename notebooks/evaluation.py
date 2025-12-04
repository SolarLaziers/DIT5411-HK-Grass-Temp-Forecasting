#%% [markdown]
# # Model Evaluation and Comparison (Enhanced)
#
# Produces clearer plots and a metrics summary file with MAE, RMSE, bias, std, R^2, and percentile error info.
# Saves figures into /figures/ and metrics into /figures/metrics_summary.txt

#%% [code]
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Resolve repo paths reliably
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Load arrays and scaler
X_test = np.load(DATA_DIR / 'X_test.npy')
y_test = np.load(DATA_DIR / 'y_test.npy')
scaler = joblib.load(DATA_DIR / 'scaler.pkl')

# Define sequence length used to align dates with X_test / y_test
seq_length = X_test.shape[1]

# Load cleaned CSV (produced by data_preprocessing.py) if available; fallback to raw CSV
CLEANED_CSV = DATA_DIR / 'daily_HKO_GMT_ALL_filled.csv'
RAW_CSV = DATA_DIR / 'daily_HKO_GMT_ALL.csv'
csv_path = CLEANED_CSV if CLEANED_CSV.exists() else RAW_CSV
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# Ensure Year/Month/Day present or try header-detection fallback
if {'Year','Month','Day'}.issubset(df.columns):
    df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
else:
    cols = [c for c in df.columns]
    def find_col(frags):
        for c in cols:
            if any(f in c.lower() for f in frags):
                return c
        return None
    ycol = find_col(['year','年'])
    mcol = find_col(['month','月'])
    dcol = find_col(['day','日'])
    if ycol and mcol and dcol:
        df['Date'] = pd.to_datetime(df[[ycol, mcol, dcol]].rename(columns={ycol:'Year', mcol:'Month', dcol:'Day'}))
    else:
        raise RuntimeError("Cannot find Year/Month/Day columns in CSV for evaluation plotting.")

df = df.set_index('Date').sort_index()

# Determine plot_dates aligned with X_test / y_test
test_slice = df['2025-01-01':'2025-10-30']
if test_slice.empty:
    all_dates = df.index
    if len(all_dates) >= len(y_test) + seq_length:
        plot_dates = all_dates[-(len(y_test)+seq_length):][seq_length:]
    else:
        plot_dates = pd.date_range(end=all_dates[-1], periods=len(y_test))
else:
    plot_dates = test_slice.index[seq_length:seq_length+len(y_test)]

# Load models (Keras format)
rnn_model = tf.keras.models.load_model(MODELS_DIR / 'rnn_model.keras')
lstm_model = tf.keras.models.load_model(MODELS_DIR / 'lstm_model.keras')

# Predictions and inverse transform
rnn_pred_scaled = rnn_model.predict(X_test)
lstm_pred_scaled = lstm_model.predict(X_test)
rnn_pred = scaler.inverse_transform(rnn_pred_scaled.reshape(-1,1)).flatten()
lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1,1)).flatten()
y_test_orig = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

# Compute metrics
errors_rnn = y_test_orig - rnn_pred
errors_lstm = y_test_orig - lstm_pred

def compute_summary(name, y_true, y_pred, errors):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(errors)            # mean error
    err_std = np.std(errors, ddof=1)
    r2 = r2_score(y_true, y_pred)
    pct_95 = np.percentile(np.abs(errors), 95)
    median_abs = np.median(np.abs(errors))
    return dict(name=name, MAE=mae, RMSE=rmse, Bias=bias, Std=err_std, R2=r2, MedianAbs=median_abs, Pct95Abs=pct_95)

summary_rnn = compute_summary('RNN', y_test_orig, rnn_pred, errors_rnn)
summary_lstm = compute_summary('LSTM', y_test_orig, lstm_pred, errors_lstm)

# Print and save metrics summary
metrics_df = pd.DataFrame([summary_rnn, summary_lstm]).set_index('name')
print(metrics_df.round(3))
(metrics_path := FIGURES_DIR / 'metrics_summary.txt').write_text(metrics_df.to_string(float_format=lambda x: f"{x:.3f}"))
print(f"Saved metrics summary to: {metrics_path}")

# Helper: rolling mean for smoother lines (window in days)
def rolling(arr, window=7):
    return pd.Series(arr, index=plot_dates).rolling(window=window, min_periods=1, center=True).mean().values

rnn_roll = rolling(rnn_pred, 7)
lstm_roll = rolling(lstm_pred, 7)
actual_roll = rolling(y_test_orig, 7)

# 1) Clean combined time-series plot with smoothed lines + shaded error envelopes
# set plotting style with robust fallback (some environments lack seaborn style)
try:
    plt.style.use('seaborn-whitegrid')
except Exception:
    available = plt.style.available
    for candidate in ('seaborn', 'seaborn-v0_8-darkgrid', 'ggplot', 'bmh', 'classic'):
        if candidate in available:
            plt.style.use(candidate)
            break
    else:
        plt.style.use('default')

plt.figure(figsize=(14,6))
plt.plot(plot_dates, y_test_orig, label='Actual', color='black', linewidth=1.0, alpha=0.9)
plt.plot(plot_dates, rnn_roll, label=f'RNN (7d MA) — MAE {summary_rnn["MAE"]:.2f}', color='C0', linewidth=1.5)
plt.plot(plot_dates, lstm_roll, label=f'LSTM (7d MA) — MAE {summary_lstm["MAE"]:.2f}', color='C1', linewidth=1.5)

# shaded error envelopes (±1 std of errors)
plt.fill_between(plot_dates, rnn_roll - np.std(errors_rnn), rnn_roll + np.std(errors_rnn), color='C0', alpha=0.12)
plt.fill_between(plot_dates, lstm_roll - np.std(errors_lstm), lstm_roll + np.std(errors_lstm), color='C1', alpha=0.12)

# annotate statistics box
textstr = (
    f"RNN: MAE={summary_rnn['MAE']:.2f}, RMSE={summary_rnn['RMSE']:.2f}, Bias={summary_rnn['Bias']:.2f}\n"
    f"LSTM: MAE={summary_lstm['MAE']:.2f}, RMSE={summary_lstm['RMSE']:.2f}, Bias={summary_lstm['Bias']:.2f}\n"
    f"95% error (abs): RNN={summary_rnn['Pct95Abs']:.2f}, LSTM={summary_lstm['Pct95Abs']:.2f}"
)
plt.gca().text(0.01, 0.01, textstr, transform=plt.gca().transAxes, fontsize=9,
               bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))
plt.legend(loc='upper left')
plt.title('Actual vs Smoothed Predictions (7-day MA) — shaded ±1 std envelopes')
plt.xlabel('Date'); plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'comparison_smoothed.png', dpi=150)
plt.show()

# 2) Error time-series with mean and ±1 std lines
plt.figure(figsize=(14,4))
plt.plot(plot_dates, errors_rnn, label='RNN error', color='C0', alpha=0.6)
plt.plot(plot_dates, errors_lstm, label='LSTM error', color='C1', alpha=0.6)
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.axhline(summary_rnn['Bias'], color='C0', linestyle=':', label=f'RNN bias {summary_rnn["Bias"]:.2f}')
plt.axhline(summary_lstm['Bias'], color='C1', linestyle=':', label=f'LSTM bias {summary_lstm["Bias"]:.2f}')
plt.fill_between(plot_dates, summary_rnn['Bias']-summary_rnn['Std'], summary_rnn['Bias']+summary_rnn['Std'], color='C0', alpha=0.08)
plt.fill_between(plot_dates, summary_lstm['Bias']-summary_lstm['Std'], summary_lstm['Bias']+summary_lstm['Std'], color='C1', alpha=0.08)
plt.legend()
plt.title('Prediction Error Over Time (Actual - Predicted)')
plt.xlabel('Date'); plt.ylabel('Error (°C)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'error_time_series.png', dpi=150)
plt.show()

# 3) Scatter plot with regression fit and R^2
plt.figure(figsize=(6,6))
plt.scatter(y_test_orig, rnn_pred, s=12, alpha=0.5, color='C0', label='RNN')
plt.scatter(y_test_orig, lstm_pred, s=12, alpha=0.5, color='C1', label='LSTM')
mn = min(y_test_orig.min(), rnn_pred.min(), lstm_pred.min())
mx = max(y_test_orig.max(), rnn_pred.max(), lstm_pred.max())
plt.plot([mn,mx], [mn,mx], 'k--', linewidth=0.8)
# regression fits
coef_rnn = np.polyfit(y_test_orig, rnn_pred, 1)
coef_lstm = np.polyfit(y_test_orig, lstm_pred, 1)
yr = np.linspace(mn, mx, 100)
plt.plot(yr, np.polyval(coef_rnn, yr), color='C0', linestyle=':', linewidth=1)
plt.plot(yr, np.polyval(coef_lstm, yr), color='C1', linestyle=':', linewidth=1)
plt.xlabel('Actual (°C)'); plt.ylabel('Predicted (°C)')
plt.title(f'Predicted vs Actual — R2 RNN={summary_rnn["R2"]:.3f}, LSTM={summary_lstm["R2"]:.3f}')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pred_vs_actual_combined.png', dpi=150)
plt.show()

# 4) Overlapped histogram + smoothed density of errors
def smooth_hist_line(values, bins=60, smooth=3):
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:])/2
    # simple convolution smoothing
    kernel = np.ones(smooth)/smooth
    hist_smooth = np.convolve(hist, kernel, mode='same')
    return centers, hist_smooth

cent_r, hist_r = smooth_hist_line(errors_rnn, bins=60, smooth=5)
cent_l, hist_l = smooth_hist_line(errors_lstm, bins=60, smooth=5)

plt.figure(figsize=(8,4))
plt.fill_between(cent_r, hist_r, color='C0', alpha=0.4, label='RNN error density')
plt.fill_between(cent_l, hist_l, color='C1', alpha=0.4, label='LSTM error density')
plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
plt.xlabel('Error (°C)'); plt.ylabel('Density')
plt.title('Overlapped Error Densities')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'error_density_overlap.png', dpi=150)
plt.show()

# 5) Save combined summary CSV (for programmatic use)
metrics_csv = FIGURES_DIR / 'metrics_summary.csv'
metrics_df.to_csv(metrics_csv)
print(f"Saved metrics CSV to: {metrics_csv}")

# End