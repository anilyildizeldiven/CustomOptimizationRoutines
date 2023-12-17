
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


class NewtonOptimizedModel(Model):
    def __init__(self):
        super(NewtonOptimizedModel, self).__init__()
        self.dense = Dense(5, activation='relu', input_shape=(4,))  # Only one hidden layer
        self.output_layer = Dense(3, activation='softmax')  # Output layer
        self.eps = tf.Variable(100.0, trainable=False)
        
    def call(self, inputs):
        x = self.dense(inputs)
        return self.output_layer(x)

    def train_step(self, data):
        # Unpack the data
        x, y = data
        
        # Subsampling - choose a random subset for this step
        batch_size = 40  # Set your desired batch size
        data_size = tf.shape(x)[0]  # Get the size of the data along the first axis
        idx = tf.random.uniform((batch_size,), 0, data_size, dtype=tf.int32)
        x_batch = tf.gather(x, idx)
        y_batch = tf.gather(y, idx)

        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape(persistent=True) as t1:
                y_pred = self(x_batch, training=True)
                loss = self.compiled_loss(y_batch, y_pred)

            # Gradient for all variables
            grads = t1.gradient(loss, self.trainable_variables)

        hessians = []
        total_size = 0
        for grad, var in zip(grads, self.trainable_variables):
            hessian = t2.jacobian(grad, var)
            hessian_flat = tf.reshape(hessian, [tf.size(var), -1])
            hessians.append(hessian_flat)
            total_size += tf.size(var)

        # Create a block-diagonal matrix from Hessians
        flat_hessians = tf.zeros([total_size, total_size], dtype=tf.float32)
        start = 0
        for hessian in hessians:
            size = hessian.shape[0]
            indices = tf.range(start, start + size)
            flat_hessians = tf.tensor_scatter_nd_update(
                flat_hessians, 
                tf.stack([indices, indices], axis=1), 
                tf.linalg.diag_part(hessian)
            )
            start += size

        # Flatten gradients
        flat_grads = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
        
        hessian_sum = tf.reduce_sum(tf.abs(flat_hessians))
        # Passe eps basierend auf den Eigenschaften der Hesse-Matrix an
        threshold = 1e-3  # Wähle einen Schwellenwert für die Hesse-Matrix

        # Condition for updating epsilon
        def condition_low():
            return self.eps * 0.9

        def condition_high():
            return self.eps * 1.1

        self.eps.assign(tf.cond(hessian_sum < threshold, condition_low, condition_high))

        # Regularize and compute the inverse Hessian
        eye_eps = tf.eye(total_size) * self.eps
        inv_hessian = tf.linalg.inv(flat_hessians + eye_eps)
    
        # Compute update
        update = tf.linalg.matmul(inv_hessian, tf.expand_dims(flat_grads, -1))
    
        # Update all variables
        start = 0
        for var in self.trainable_variables:
            size = tf.size(var)
            var_update = tf.reshape(update[start:start + size], var.shape)
            var.assign_sub(var_update)
            start += size
    
        del t1, t2
        self.compiled_metrics.update_state(y_batch, y_pred)
        return {m.name: m.result() for m in self.metrics}
        



    
    
file_path = '/Users/anilcaneldiven/Desktop/iris.csv'
data = pd.read_csv(file_path)


X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

#  class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

# Convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=42)


class StoreWeightNormCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(StoreWeightNormCallback, self).__init__()
        self.weight_norms_per_epoch = []

    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        weight_norms = [np.linalg.norm(w) for w in weights]
        self.weight_norms_per_epoch.append(weight_norms)
        print(f"Epoch {epoch + 1}, Weight norms: {weight_norms}")

        
        
# Create and compile model
model = NewtonOptimizedModel()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
#Wich other metrices?


# Train the model with the custom callback

callback = StoreWeightNormCallback()


model.fit(X_train, y_train, batch_size=30, epochs=10, validation_split=0.2, callbacks=[callback])

# After training, access the stored weight norms
epoch_weight_norms = callback.weight_norms_per_epoch

# You can now print or analyze epoch_weight_norms
print(epoch_weight_norms)

# Evaluate the model
scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose="auto")
print(f"Accuracy: {scores[1]*100}")