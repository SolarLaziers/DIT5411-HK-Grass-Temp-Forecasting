#!/usr/bin/env python3
#%% [markdown]
# # Data Preprocessing for HK Grass Min Temperature Forecasting
#
# This notebook-like script (for use in VS Code / Jupyter interactive cells):
# - Detects bilingual header and skips leading description lines
# - Cleans the Value column (removes trailing letters like 'C', maps '***' to NaN)
# - Interpolates missing values (time-aware), then ffill/bfill for edges
# - Saves a cleaned CSV (data/daily_HKO_GMT_ALL_filled.csv), numpy arrays and scaler
# - Creates train/test sequences for RNN/LSTM
#
# Run cells sequentially in a Jupyter environment or execute the whole script.

#%% [code]
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

#%% [code]
# Paths and directories
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
CSV_PATH = DATA_DIR / 'daily_HKO_GMT_ALL.csv'
CLEANED_CSV_PATH = DATA_DIR / 'daily_HKO_GMT_ALL_filled.csv'
OUT_DIR = DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

#%% [code]
# Ensure CSV exists
if not CSV_PATH.exists():
    available = [p.name for p in DATA_DIR.glob('*')]
    raise FileNotFoundError(
        f"{CSV_PATH} not found. Place the CSV in {DATA_DIR}.\n"
        f"Files currently in {DATA_DIR}: {available}"
    )

#%% [code]
def _read_csv_with_header_detection(path, encoding='utf-8-sig'):
    """
    Detect header row by scanning the file for a line that looks like
    a header (contains commas and tokens like 'Year'/'年'/'月').
    Returns a pandas DataFrame with that header.
    """
    header_idx = None
    with open(path, encoding=encoding, errors='replace') as f:
        for i, line in enumerate(f):
            if ',' in line and any(tok in line.lower() for tok in ('year', '年', '月', 'month')):
                header_idx = i
                break
    if header_idx is None:
        header_idx = 0
    df = pd.read_csv(path, encoding=encoding, skiprows=header_idx)
    return df

#%% [code]
# Read raw CSV and normalize headers
df = _read_csv_with_header_detection(CSV_PATH)
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

#%% [code]
# Map possible column names to canonical names
col_map = {}
for c in df.columns:
    if not isinstance(c, str):
        continue
    lower = c.lower()
    if 'year' in lower or '年' in lower:
        col_map[c] = 'Year'
    elif 'month' in lower or '月' in lower:
        col_map[c] = 'Month'
    elif 'day' in lower or ('日' in lower and 'value' not in lower):
        col_map[c] = 'Day'
    elif 'value' in lower or '數值' in lower or '值' in lower:
        col_map[c] = 'Value'
    elif 'completeness' in lower or '完整' in lower:
        col_map[c] = 'Completeness'

df = df.rename(columns=col_map)

#%% [code]
# Verify required columns
for req in ('Year', 'Month', 'Day', 'Value'):
    if req not in df.columns:
        raise KeyError(f"Required column '{req}' not found in CSV headers: {list(df.columns)}")

#%% [code]
# Clean Value column:
# - Convert to str, strip whitespace
# - Treat '***' and blanks as NaN
# - Remove non-numeric characters (e.g. trailing 'C')
# - Convert to numeric
val_raw = df['Value'].astype(str).str.strip()
val_raw = val_raw.replace({'nan': np.nan, '***': np.nan, '': np.nan})
val_clean = val_raw.str.replace(r'[^0-9\.\-]', '', regex=True)
val_clean = val_clean.replace('', np.nan)
df['Value'] = pd.to_numeric(val_clean, errors='coerce')

#%% [code]
# Build datetime index robustly (coerce invalid values)
df['Date'] = pd.to_datetime(
    dict(
        year=pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int),
        month=pd.to_numeric(df['Month'], errors='coerce').fillna(1).astype(int),
        day=pd.to_numeric(df['Day'], errors='coerce').fillna(1).astype(int)
    ),
    errors='coerce'
)

# Drop rows with invalid dates, set index
df = df[~df['Date'].isna()].copy()
df = df.set_index('Date').sort_index()
df = df[['Value'] + [c for c in df.columns if c not in ['Value']]]

#%% [code]
# Mark missing values prior to filling
df['WasMissing'] = df['Value'].isna()

#%% [code]
# Interpolate missing values (time-based if possible), then ffill/bfill for edges
try:
    df['Value'] = df['Value'].interpolate(method='time', limit_direction='both')
except Exception:
    df['Value'] = df['Value'].interpolate(limit_direction='both')
df['Value'] = df['Value'].ffill().bfill()

filled_count = int(df['WasMissing'].sum())
total_count = len(df)
print(f"Missing values found and filled: {filled_count} / {total_count}")

#%% [code]
# Round to 1 decimal to match original format and save cleaned CSV
df['Value'] = df['Value'].round(1)

df_out = df.copy()
df_out['WasFilled'] = df_out['WasMissing']
df_out['Year'] = df_out.index.year
df_out['Month'] = df_out.index.month
df_out['Day'] = df_out.index.day
cols_order = ['Year', 'Month', 'Day', 'Value', 'WasFilled']
if 'Completeness' in df.columns:
    cols_order.append('Completeness')
df_out.to_csv(CLEANED_CSV_PATH, index=False, float_format='%.1f')
print(f"Cleaned CSV with filled values saved to: {CLEANED_CSV_PATH}")

#%% [code]
# Create train/test splits (use requested ranges; fallback to last-10% split)
DEFAULT_TRAIN_START = '1980-01-01'
DEFAULT_TRAIN_END = '2024-12-31'
DEFAULT_TEST_START = '2025-01-01'
DEFAULT_TEST_END = '2025-10-30'

train_df = df[DEFAULT_TRAIN_START:DEFAULT_TRAIN_END]['Value']
test_df = df[DEFAULT_TEST_START:DEFAULT_TEST_END]['Value']

if train_df.empty or test_df.empty:
    n = len(df)
    if n < 50:
        raise ValueError("Not enough data after cleaning to create train/test splits.")
    split = int(n * 0.9)
    train_df = df.iloc[:split]['Value']
    test_df = df.iloc[split:]['Value']
    print(f"Using fallback split. Train size: {len(train_df)}, Test size: {len(test_df)}")
else:
    print(f"Using requested date slices. Train size: {len(train_df)}, Test size: {len(test_df)}")

#%% [code]
# Scale data and save scaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df.values.reshape(-1, 1))
test_scaled = scaler.transform(test_df.values.reshape(-1, 1))
joblib.dump(scaler, OUT_DIR / 'scaler.pkl')

#%% [code]
# Sequence creation for RNN/LSTM
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

if X_train.size == 0 or X_test.size == 0:
    raise ValueError("Empty sequences after sequence creation. Reduce seq_length or provide more data.")

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

#%% [code]
# Save processed arrays
np.save(OUT_DIR / 'X_train.npy', X_train)
np.save(OUT_DIR / 'y_train.npy', y_train)
np.save(OUT_DIR / 'X_test.npy', X_test)
np.save(OUT_DIR / 'y_test.npy', y_test)

print(f"Saved numpy arrays and scaler in {OUT_DIR}")

#%% [code]
# Optional visualization (run interactively)
if __name__ == '__main__':
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Value'], label='Value (filled)')
    plt.scatter(df.index[df['WasMissing']], df.loc[df['WasMissing'], 'Value'], color='red', s=10, label='Originally missing (filled)')
    plt.title('Historical Grass Min Temperature (filled)')
    plt.legend()
    plt.show()
