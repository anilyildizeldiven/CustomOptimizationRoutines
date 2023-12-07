#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:47:35 2023

@author: anilcaneldiven
"""
class NewtonOptimizedModel(Model):
    def __init__(self):
        super(NewtonOptimizedModel, self).__init__()
        self.learning_rate = 0.001
        self.epsilon = 1e-3
        self.batch_size = X_train.shape[0]
        self.layers_list = [
            Dense(8, activation='tanh', input_shape=(4,)),
            Dense(8, activation='tanh'),
            Dense(3, activation='softmax')
        ]
    
    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x
    
    def asymmetric_case(self, h_mat):
        return 0.5 * (h_mat + tf.transpose(h_mat))

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape() as t1:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred)
            
            for layer in self.layers_list:
                if hasattr(layer, 'kernel'):
                    g = t1.gradient(loss, layer.kernel)
                    
                    if g is not None:
                        h = t2.jacobian(g, layer.kernel)
                        
                        #print("Hessematrix:")
                        #print(h) 
                        
                        n_params = tf.reduce_prod(layer.kernel.shape)
                        g_vec = tf.reshape(g, [n_params, 1])
                        h_mat = tf.reshape(h, [n_params, n_params])
                        eps = self.epsilon
                        eye_eps = tf.eye(h_mat.shape[0]) * eps
                        update = tf.linalg.solve(h_mat + eye_eps, g_vec)
                        
                        #print("Update:")
                        #print(update)
                        
                        layer.kernel.assign_sub(tf.reshape(update, layer.kernel.shape))
                
                del t2
                self.compiled_metrics.update_state(y, y_pred)
                return {m.name: m.result() for m in self.metrics}
            
   file_path = '/Users/anilcaneldiven/Desktop/iris.csv'
   data = pd.read_csv(file_path)
    X = data.iloc[:, 0:4].values
    y = data.iloc[:, 4].values
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
   dummy_y = to_categorical(encoded_Y)
   X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=42)
    model = NewtonOptimizedModel()
   model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
  batch_size = X_train.shape[0]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=100, verbose=1, validation_split=0.2)
    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose="auto")
   print(f"Accuracy: {scores[1]*100}")


#################################################################################
   
#################################################################################


class TestNewtonOptimizedModel(unittest.TestCase):
    def setUp(self):
        # Initialize variables for testing
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
        
        # Train the model
        batch_size = X_train.shape[0]
        epochs = 10  # You can adjust the number of epochs
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.2)

    def test_model_initialization(self):
        self.assertEqual(self.model.learning_rate, 0.001)
        self.assertEqual(self.model.epsilon, 1e-3)
        # Add more assertions based on your model attributes
        
    def test_forward_pass(self):
        input_data = tf.constant(self.X_test, dtype=tf.float32)
        output = self.model.call(input_data)
        expected_shape = (input_data.shape[0], 3)  # Assuming the output shape is (batch_size, 3)
        self.assertEqual(output.shape, expected_shape)

    def test_train_step(self):
        # Wandle die Daten in TensorFlow-kompatible Tensoren um
        X_train_tf = tf.constant(self.X_test, dtype=tf.float32)
        y_train_tf = tf.constant(self.y_test, dtype=tf.float32)
        
        # Mock training step
        data = (X_train_tf, y_train_tf)
        result = self.model.train_step(data)

        # Validate the output of train_step method
        # Assuming the result dictionary returns loss and metrics
        self.assertIn('loss', result)
        self.assertIn('accuracy', result)  # Assuming 'accuracy' is a metric used

    def test_model_compilation(self):
        self.assertTrue(hasattr(self.model, 'compiled_loss'))
        self.assertTrue(hasattr(self.model, 'compiled_metrics'))
        # Add more assertions to validate compilation
        
    def test_hyperparameter_updates(self):
        initial_learning_rate = self.model.learning_rate
        initial_epsilon = self.model.epsilon

        # Change hyperparameters
        new_learning_rate = 0.01
        new_epsilon = 1e-4
        self.model.learning_rate = new_learning_rate
        self.model.epsilon = new_epsilon

        self.assertNotEqual(self.model.learning_rate, initial_learning_rate)
        self.assertNotEqual(self.model.epsilon, initial_epsilon)

    def test_weights_update(self):
        initial_weights = [layer.kernel.numpy() for layer in self.model.layers_list]

        # Perform additional training steps
        # You can define more steps or epochs as needed
        X_test_tf = tf.constant(self.X_test, dtype=tf.float32)
        y_test_tf = tf.constant(self.y_test, dtype=tf.float32)
        self.model.train_step((X_test_tf, y_test_tf))

        updated_weights = [layer.kernel.numpy() for layer in self.model.layers_list]
        for initial_w, updated_w in zip(initial_weights, updated_weights):
            self.assertFalse(np.array_equal(initial_w, updated_w))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    
