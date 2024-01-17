#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:12:03 2023

@author: anilcaneldiven
"""
import sys
sys.path.append('/Users/anilcaneldiven/Desktop/python_work/')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import get_custom_objects
import numpy as np
from sklearn.preprocessing import StandardScaler
# Annahme: NewtonOptimizer ist deine benutzerdefinierte Optimiererklasse
from customOptimizer import NewtonOptimizer  # Stelle sicher, dass du den korrekten Pfad ersetzt
import matplotlib.pyplot as plt
# Registriere deinen benutzerdefinierten Optimierer in get_custom_objects
get_custom_objects().update({'NewtonOptimizer': NewtonOptimizer})

# Dummy-Daten erstellen
np.random.seed(42)
X_train = np.random.rand(100, 5)  # 100 Datenpunkte mit 5 Features
y_train = np.random.rand(100, 1)  # Zugehörige Zielwerte

# Ein einfaches Sequential-Modell erstellen
model = Sequential()
model.add(Dense(10, input_shape=(5,), activation='LeakyReLU',kernel_regularizer=regularizers.l2(1)))  # Dense-Schicht mit 10 Neuronen
model.add(Dense(10, activation='LeakyReLU',kernel_regularizer=regularizers.l2(1)))  # Dense-Schicht mit 10 Neuronen
model.add(Dense(10, activation='LeakyReLU',kernel_regularizer=regularizers.l2(1)))  # Dense-Schicht mit 10 Neuronen
model.add(Dense(1, activation='linear'))  # Output-Schicht mit 1 Neuron (lineare Aktivierung)


# NewtonOptimizer initialisieren und Gewichte des Modells setzen
optimizer = NewtonOptimizer()
optimizer.initialize_weights(model)

# Model kompilieren
model.compile(optimizer='NewtonOptimizer', loss='mae')

class StoreWeightNormCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(StoreWeightNormCallback, self).__init__()
        self.weight_norms_per_epoch = []

    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        weight_norms = [np.linalg.norm(w) for w in weights]
        self.weight_norms_per_epoch.append(weight_norms)
        print(f"Epoch {epoch + 1}, Weight norms: {weight_norms}")
        
# Modell trainieren
callback = StoreWeightNormCallback()

history = model.fit(X_train, y_train, batch_size=X_train.shape[0], epochs=100, validation_split=0.2, callbacks=[callback])

# Plot Loss-Verlauf
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plt.hist(y_train, bins=20)
# plt.xlabel('y_train')
# plt.ylabel('Frequency')
# plt.title('Histogram of y_train')
# plt.show()



