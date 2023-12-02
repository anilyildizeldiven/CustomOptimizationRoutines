import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import pandas as pd

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
#https://stackoverflow.com/questions/36697736/how-to-force-tensorflow-tensors-to-be-symmetric
    def asymmetric_case(self, h_mat):
        return 0.5 * (h_mat + tf.transpose(h_mat))

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape() as t1:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred)
            g = t1.gradient(loss, self.dense1.kernel)
            if g is None:
                # Approximation des Gradienten mit tfp.math.value_and_gradient => finite difference method
                g = tfp.math.value_and_gradient(lambda var: self.compiled_loss(y, self(x)), self.dense1.kernel)[1]

            h = t2.jacobian(g, self.dense1.kernel)

            n_params = tf.reduce_prod(self.dense1.kernel.shape)
            g_vec = tf.reshape(g, [n_params, 1])
            h_mat = tf.reshape(h, [n_params, n_params])

            is_symmetric = tf.reduce_all(tf.equal(h_mat, tf.transpose(h_mat)))
            # ist eine if then klausel in tensorflow
            #https://stackoverflow.com/questions/35833011/how-to-add-if-condition-in-a-tensorflow-graph
            h_mat = tf.cond(is_symmetric, lambda: self.asymmetric_case(h_mat), lambda: h_mat)

            eye_eps = tf.eye(h_mat.shape[0]) * self.epsilon
            
            update = tf.linalg.solve(h_mat + eye_eps, g_vec)
            self.dense1.kernel.assign_sub(tf.reshape(update, self.dense1.kernel.shape))

        del t2
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
