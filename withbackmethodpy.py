#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:06:27 2023

@author: anilcaneldiven
"""
#https://gist.github.com/jgomezdans/3144636
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

    def hessian(self, x0, epsilon=1.e-5, *args):
        f1 = approx_fprime(x0, calculate_cost_function, *args)

        n = x0.shape[0]
        hessian_matrix = np.zeros((n, n))

        for j in range(n):
            xx0 = x0[j]
            x0[j] = xx0 + epsilon
            f2 = approx_fprime(x0, calculate_cost_function, *args)
            hessian_matrix[:, j] = (f2 - f1) / epsilon
            x0[j] = xx0

        return hessian_matrix

    def approximate_gradient(self, x, y):
        x_np = x.numpy()
        hessian_matrix = self.hessian(x_np)
        approx_grad = np.dot(hessian_matrix, x_np)
        return tf.constant(approx_grad)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape() as t1:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred)
            g = t1.gradient(loss, self.dense1.kernel)

            if g is None:
                g = self.approximate_gradient(x, y)

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



