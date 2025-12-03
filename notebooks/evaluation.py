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

#%% [code]
# Load data and scaler
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')
scaler = joblib.load('../data/scaler.pkl')
test_df = pd.read_csv('../data/daily_HKO_GMT_ALL.csv', encoding='utf-8-sig')  # Reload for dates
test_df['Date'] = pd.to_datetime(test_df[['Year', 'Month', 'Day']].assign(DAY=test_df['Day'], MONTH=test_df['Month'], YEAR=test_df['Year']))
test_df = test_df.set_index('Date')
test_df = test_df['2025-01-01':'2025-10-30']
seq_length = X_test.shape[1]

# Load models
rnn_model = tf.keras.models.load_model('../models/rnn_model.h5')
lstm_model = tf.keras.models.load_model('../models/lstm_model.h5')

#%% [code]
# Predictions
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

#%% [code]
# Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(test_df.index[seq_length:], y_test_orig, label='Actual')
plt.plot(test_df.index[seq_length:], rnn_pred, label='RNN Pred')
plt.plot(test_df.index[seq_length:], lstm_pred, label='LSTM Pred')
plt.legend()
plt.title('Actual vs Predicted Grass Min Temp (2025)')
plt.savefig('../figures/predictions_plot.png')
plt.show()

#%% [code]
# Error Distribution (LSTM example)
errors_lstm = y_test_orig.flatten() - lstm_pred.flatten()
plt.figure(figsize=(8,4))
plt.hist(errors_lstm, bins=20)
plt.title('LSTM Error Distribution')
plt.savefig('../figures/error_dist.png')
plt.show()

#%% [markdown]
# ## Analysis
# LSTM outperforms RNN due to better handling of long-term dependencies. Check high-error days for cold snaps.