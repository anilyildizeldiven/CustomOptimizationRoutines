import sys
sys.path.append('/Users/anilcaneldiven/Desktop/python_work/')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import get_custom_objects
import numpy as np

# Annahme: NewtonOptimizer ist deine benutzerdefinierte Optimiererklasse
from customOptimizer import NewtonOptimizer 

# Registriere deinen benutzerdefinierten Optimierer in get_custom_objects
get_custom_objects().update({'NewtonOptimizer': NewtonOptimizer})

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ein einfaches Sequential-Modell erstellen
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu')) 
model.add(Dense(10, activation='LeakyReLU'))
model.add(Dense(10, activation='LeakyReLU'))
model.add(Dense(10, activation='LeakyReLU'))
 # Anpassen der Eingabeform, falls nötig
model.add(Dense(3, activation='softmax'))  # Output-Schicht mit 3 Neuronen für Iris-Datensatz (Klassen)

# CustomOptimizer initialisieren und Gewichte des Modells setzen
optimizer = NewtonOptimizer()  # Hier nutzt du deinen eigenen Optimizer
optimizer.initialize_weights(model)

# Model kompilieren
model.compile(optimizer='NewtonOptimizer', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callback für Gewichtsnormen
callback = StoreWeightNormCallback()

# Modell trainieren
model.fit(X_train, y_train, batch_size=50, epochs=100, validation_split=0.2, callbacks=[callback])

# Evaluierung auf dem Testset
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")



