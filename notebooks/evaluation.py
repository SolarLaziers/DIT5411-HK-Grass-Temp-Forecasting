#%% [markdown]
# # Model Evaluation and Comparison
#
# This script evaluates RNN and LSTM on test data, computes metrics, and generates visualizations.
#
# **Input**: Processed test data and trained models
#
# **Output**: Metrics and plots in /figures/

#%% [code]
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from pathlib import Path

# Resolve repo paths reliably
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load arrays and scaler
X_test = np.load(DATA_DIR / 'X_test.npy')
y_test = np.load(DATA_DIR / 'y_test.npy')
scaler = joblib.load(DATA_DIR / 'scaler.pkl')

# Load cleaned CSV (produced by data_preprocessing.py) if available; fallback to raw CSV
CLEANED_CSV = DATA_DIR / 'daily_HKO_GMT_ALL_filled.csv'
RAW_CSV = DATA_DIR / 'daily_HKO_GMT_ALL.csv'
csv_path = CLEANED_CSV if CLEANED_CSV.exists() else RAW_CSV
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# Ensure Year/Month/Day present or try header-detection fallback
if {'Year','Month','Day'}.issubset(df.columns):
    df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
else:
    # Attempt to detect header and rebuild Date if columns are bilingual
    cols = [c for c in df.columns]
    # try to find columns by name fragments
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

# Select test period (if cleaned CSV contains full range, slicing will work)
test_slice = df['2025-01-01':'2025-10-30']
if test_slice.empty:
    # If slice empty, attempt to align using length of y_test and last dates
    all_dates = df.index
    # dates corresponding to y_test start at index seq_length within the test_df used by preprocessing
    seq_length = X_test.shape[1]
    # assume test sequences were created from contiguous test_df used in preprocessing; pick last len(y_test) dates
    if len(all_dates) >= len(y_test) + seq_length:
        plot_dates = all_dates[-(len(y_test)+seq_length):][seq_length:]
    else:
        # fallback to integer range
        plot_dates = pd.date_range(end=all_dates[-1], periods=len(y_test))
else:
    plot_dates = test_slice.index[seq_length:seq_length+len(y_test)]

# Load models (Keras format)
rnn_model = tf.keras.models.load_model(MODELS_DIR / 'rnn_model.keras')
lstm_model = tf.keras.models.load_model(MODELS_DIR / 'lstm_model.keras')

# Predictions and inverse transform
rnn_pred_scaled = rnn_model.predict(X_test)
rnn_pred = scaler.inverse_transform(rnn_pred_scaled)
y_test_orig = scaler.inverse_transform(y_test.reshape(-1,1))

lstm_pred_scaled = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

# Metrics
mae_rnn = mean_absolute_error(y_test_orig, rnn_pred)
rmse_rnn = np.sqrt(mean_squared_error(y_test_orig, rnn_pred))
mae_lstm = mean_absolute_error(y_test_orig, lstm_pred)
rmse_lstm = np.sqrt(mean_squared_error(y_test_orig, lstm_pred))

print(f"RNN: MAE={mae_rnn:.2f}, RMSE={rmse_rnn:.2f}")
print(f"LSTM: MAE={mae_lstm:.2f}, RMSE={rmse_lstm:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(plot_dates, y_test_orig, label='Actual')
plt.plot(plot_dates, rnn_pred, label='RNN Pred')
plt.plot(plot_dates, lstm_pred, label='LSTM Pred')
plt.legend()
plt.title('Actual vs Predicted Grass Min Temp (Test Period)')
plt.savefig(FIGURES_DIR / 'predictions_plot.png')
plt.show()

# Error Distribution (LSTM example)
errors_lstm = y_test_orig.flatten() - lstm_pred.flatten()
plt.figure(figsize=(8,4))
plt.hist(errors_lstm, bins=20)
plt.title('LSTM Error Distribution')
plt.savefig(FIGURES_DIR / 'error_dist.png')
plt.show()

#%% [markdown]
# ## Analysis
# LSTM outperforms RNN due to better handling of long-term dependencies. Check high-error days for cold snaps.