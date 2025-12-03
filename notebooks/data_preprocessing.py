#%% [markdown]
# # Data Preprocessing for HK Grass Min Temperature Forecasting
#
# This script handles data loading, cleaning, subsetting, scaling, and sequence creation.
#
# **Dataset**: daily_HKO_GMT_ALL.csv (place in /data/ folder)
#
# **Output**: Prepared sequences for training RNN/LSTM.

#%% [code]
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import joblib  # For saving scaler

#%% [code]
# Load the CSV
df = pd.read_csv('../data/daily_HKO_GMT_ALL.csv', encoding='utf-8-sig')

# Replace "***" with NaN and convert Value to float
df['Value'] = df['Value'].replace('***', np.nan).astype(float)

# Handle missing values (forward fill)
df['Value'] = df['Value'].fillna(method='ffill')

# Create datetime column
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']].assign(DAY=df['Day'], MONTH=df['Month'], YEAR=df['Year']))
df = df.set_index('Date')
df = df[['Value']]  # Keep only temperature

# Subset training and testing
train_df = df['1980-01-01':'2024-12-31']
test_df = df['2025-01-01':'2025-10-30']

#%% [code]
# Scale the data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# Function to create sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Create sequences
seq_length = 30
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Reshape for RNN/LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

#%% [code]
# Save processed data
np.save('../data/X_train.npy', X_train)
np.save('../data/y_train.npy', y_train)
np.save('../data/X_test.npy', X_test)
np.save('../data/y_test.npy', y_test)
joblib.dump(scaler, '../data/scaler.pkl')

#%% [markdown]
# ## Visualization (Optional)
# Plot the raw data to check trends.

#%% [code]
df.plot(figsize=(12,6))
plt.title('Historical Grass Min Temperature')
plt.show()