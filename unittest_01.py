get_custom_objects().update({'NewtonOptimizer': NewtonOptimizer})

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dummy-Daten erstellen
np.random.seed(42)
X_train = np.random.rand(100, 5)  # 100 Datenpunkte mit 5 Features
y_train = np.random.rand(100, 1)  # Zugehörige Zielwerte

# Ein einfaches Sequential-Modell erstellen
model = Sequential()
model.add(Dense(10, input_shape=(5,), activation='relu'))  # Dense-Schicht mit 10 Neuronen
model.add(Dense(1, activation='linear'))  # Output-Schicht mit 1 Neuron (lineare Aktivierung)

# Modell kompilieren
model.compile(optimizer='NewtonOptimizer', loss='mean_squared_error')

# Modell trainieren
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)