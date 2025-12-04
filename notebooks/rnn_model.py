#%% [markdown]
# # RNN Model Training
#
# This script builds and trains the baseline RNN model.
#
# **Input**: Processed data from data_preprocessing.py
#
# **Output**: Trained RNN model (.h5 file)

#%% [code]
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
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

# Build RNN
model = Sequential()
model.add(SimpleRNN(50, input_shape=(seq_length, n_features), return_sequences=False))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(MODELS_DIR / 'rnn_model.h5', monitor='val_loss', save_best_only=True)
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

# Ensure final best model saved (checkpoint handles best)
model.save(MODELS_DIR / 'rnn_model_final.h5')

# Plot training history and save figure
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('RNN Training Loss')
plt.savefig(FIGURES_DIR / 'rnn_training_loss.png')
plt.show()