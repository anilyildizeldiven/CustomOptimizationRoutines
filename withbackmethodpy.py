import numpy as np
from scipy.optimize import approx_fprime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class NewtonOptimizedModel(Model):
    def __init__(self, learning_rate=0.001, batch_size=32, epsilon=1e-3):
        super(NewtonOptimizedModel, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.dense1 = Dense(8, activation='relu', input_shape=(4,))
        self.dense2 = Dense(3, activation='softmax')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
 
    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape() as t1:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred)
            g = t1.gradient(loss, self.dense1.kernel)
            if g is None:
                # Wenn g None ist, approximiere den Gradienten mit approx_fprime
                current_weights = self.dense1.kernel.numpy()  # Aktuelle Gewichte
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred)
                g = approx_fprime(current_weights, loss, self.epsilon, *args)
                g = tf.constant(g, dtype=tf.float32)
            
            h = t2.jacobian(g, self.dense1.kernel)

            n_params = tf.reduce_prod(self.dense1.kernel.shape)
            g_vec = tf.reshape(g, [n_params, 1])
            h_mat = tf.reshape(h, [n_params, n_params])

            eye_eps = tf.eye(h_mat.shape[0]) * self.epsilon
            update = tf.linalg.solve(h_mat + eye_eps, g_vec)
            self.dense1.kernel.assign_sub(tf.reshape(update, self.dense1.kernel.shape))

        del t2
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    
    
    
    
    # Optionally call summary() and get_layer() method in Model class

    # Load dataset
    file_path = '/Users/anilcaneldiven/Desktop/iris.csv'
    data = pd.read_csv(file_path)

    # Preprocess data
    X = data.iloc[:, 0:4].values
    y = data.iloc[:, 4].values

    # Encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)

    # Convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_Y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        dummy_y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    # Create and compile model
    model = NewtonOptimizedModel(learning_rate=0.001, batch_size=64, epsilon=1e-4)
    model.compile(loss='categorical_crossentropy', 
                  metrics=['accuracy'])

# Setze die Gewichte der ersten Schicht explizit auf None, um den Gradienten zu erzwingen
model.layers[0].kernel = None

    # Set batch size
    batch_size = X_train.shape[0]

    # Train model
    model.fit(X_train, 
              y_train, 
              batch_size=batch_size, 
              epochs=20, 
              verbose=1, 
              validation_split=0.2)

    # Evaluate the model
    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose="auto")
    print(f"Accuracy: {scores[1]*100}")
