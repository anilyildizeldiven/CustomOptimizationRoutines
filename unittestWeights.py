import unittest
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

# Deine NewtonOptimizedModel-Klasse bleibt unverändert

class TestNewtonOptimizedModel(unittest.TestCase):
    def setUp(self):
        # Initialisiere Variablen für den Test
        file_path = '/Users/anilcaneldiven/Desktop/iris.csv'
        data = pd.read_csv(file_path)
        X = data.iloc[:, 0:4].values
        y = data.iloc[:, 4].values
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        dummy_y = to_categorical(encoded_Y)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=42)
        
        self.model = NewtonOptimizedModel()
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Trainiere das Modell
        batch_size = 32  # Passe die Batch-Größe an
        epochs = 5  # Passe die Anzahl der Epochen an
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.2)

    def test_weights_update(self):
        initial_weights = [tf.identity(var).numpy() for var in self.model.trainable_variables]

        # Führe einen zusätzlichen Trainingsschritt aus
        X_test_tf = tf.constant(self.X_test, dtype=tf.float32)
        y_test_tf = tf.constant(self.y_test, dtype=tf.float32)
        data = (X_test_tf, y_test_tf)
        self.model.train_step(data)

        updated_weights = [tf.identity(var).numpy() for var in self.model.trainable_variables]
        for initial_w, updated_w in zip(initial_weights, updated_weights):
            # Überprüfe, ob sich die Gewichte nach dem Trainingsschritt geändert haben
            self.assertFalse(np.array_equal(initial_w, updated_w))

if __name__ == '__main__':
    unittest.main()
