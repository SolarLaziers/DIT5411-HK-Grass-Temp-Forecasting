#%% [markdown]
# # LSTM Model Training
#
# This script builds and trains the advanced LSTM model.
#
# **Input**: Processed data from data_preprocessing.py
#
# **Output**: Trained LSTM model (.h5 file)

#%% [code]
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

#%% [code]
# Load processed data
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')
seq_length = X_train.shape[1]

# Build LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

#%% [code]
# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Save model
model.save('../models/lstm_model.h5')

#%% [code]
# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('LSTM Training Loss')
plt.show()