#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 23:29:32 2024

@author: anilcaneldiven
"""

import tensorflow as tf

# Definiere eine einfache quadratische Funktion
def quadratic_function(x):
    return tf.reduce_sum(tf.square(x - 5))

def complex_function(x):
    return tf.reduce_sum(tf.sin(x) * tf.square(x - 2))



# Optimiere diese Funktion mit beiden Optimierern
x_init = tf.Variable([10.0, 10.0])  # Startpunkt

# Dein Optimierer
optimizer_custom = TestGD()
# Standard GD Optimierer
optimizer_gd = tf.optimizers.SGD(learning_rate=0.1)

# Trainingsschleife
for step in range(100):
    with tf.GradientTape() as tape:
        loss_custom = complex_function(x_init)
    grads = tape.gradient(loss_custom, [x_init])
    optimizer_custom.apply_gradients(zip(grads, [x_init]))

    # Vergleiche mit Standard GD
    with tf.GradientTape() as tape:
        loss_gd = complex_function(x_init)
    grads = tape.gradient(loss_gd, [x_init])
    optimizer_gd.apply_gradients(zip(grads, [x_init]))

    if step % 10 == 0:
        print(f"Schritt {step}, Loss Custom: {loss_custom.numpy()}, Loss GD: {loss_gd.numpy()}")
