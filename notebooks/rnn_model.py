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
import matplotlib.pyplot as plt

#%% [code]
# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# Optional: Enable device logging to see placement
tf.debugging.set_log_device_placement(True)

#%% [code]
# Load processed data
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')
seq_length = X_train.shape[1]

# Build RNN
model = Sequential()
model.add(SimpleRNN(50, input_shape=(seq_length, 1), return_sequences=False))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

#%% [code]
# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Save model
model.save('../models/rnn_model.h5')

#%% [code]
# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('RNN Training Loss')
plt.show()