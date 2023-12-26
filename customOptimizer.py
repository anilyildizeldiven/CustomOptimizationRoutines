#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:41:34 2023

@author: anilcaneldiven
"""
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/gradient_descent.py
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


class NewtonOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=0.01, name="NewtonOptimizer", **kwargs):
        super(NewtonOptimizer, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "accumulator")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                loss = tf.reduce_mean(var**2)
    
            g = t1.gradient(loss, var)
    
        h = t2.jacobian(g, var)
    
        # Reshape des Gradienten und der Hesse-Matrix
        n_params = tf.reduce_prod(var.shape)
        g_vec = tf.reshape(g, [n_params, 1])
        h_mat = tf.reshape(h, [n_params, n_params])
    
        # Truncation der Hesse-Matrix (nur dominante Eigenwerte beibehalten)
        threshold = 1e-3  # Schwellenwert für Eigenwerte
        eigenvalues, eigenvectors = tf.linalg.eigh(h_mat)
        eigenvalues = tf.where(eigenvalues < threshold, threshold, eigenvalues)
        h_mat_truncated = tf.matmul(tf.matmul(eigenvectors, tf.linalg.diag(eigenvalues)), tf.transpose(eigenvectors))
    
        # Durchführung der Hessian-basierten Newton's Methode mit abgeschnittener Hesse-Matrix
        eps = 1e-4
        eye_eps = tf.eye(h_mat_truncated.shape[0]) * eps
        update = tf.linalg.solve(h_mat_truncated + eye_eps, g_vec)
        update *= self.learning_rate
        # Aktualisierung des Variablenwertes
        var_update = var - tf.reshape(update, var.shape)
        var.assign(var_update)
    
        return var_update

        
    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")
    
    def get_config(self):
        config = super(NewtonOptimizer, self).get_config()
        config.update({ "learning_rate": self._serialize_hyperparameter("learning_rate")})
        return config
    
    def initialize_weights(self, model):
        for var in model.trainable_variables:
            var.assign(tf.random.uniform(var.shape, -1, 1))
   