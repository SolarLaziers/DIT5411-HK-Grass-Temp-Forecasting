#%% [markdown]
# # LSTM Model Training
#
# This script builds and trains the advanced LSTM model.
#
# **Input**: Processed data from data_preprocessing.py
#
# **Output**: Trained LSTM model (.keras file)

#%% [code]
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve repo paths reliably
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

#%% load processed data
X_train = np.load(DATA_DIR / 'X_train.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')
seq_length = X_train.shape[1]
n_features = X_train.shape[2] if X_train.ndim == 3 else 1

# Build LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Callbacks (save best in Keras format)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(MODELS_DIR / 'lstm_model.keras', monitor='val_loss', save_best_only=True)
]

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Ensure final best model saved (Keras format)
model.save(MODELS_DIR / 'lstm_model_final.keras')

# Plot training history and save figure
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('LSTM Training Loss')
plt.savefig(FIGURES_DIR / 'lstm_training_loss.png')
plt.show()