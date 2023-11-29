#!/usr/bin/env python
"""
Some Hessian codes
"""
import numpy as np
from scipy.optimize import approx_fprime


def hessian ( x0, epsilon=1.e-5, linear_approx=False, *args ):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum)
    """
    # ``calculate_cost_function`` is the cost function implementation
    # The next line calculates an approximation to the first
    # derivative
    f1 = approx_fprime( x0, calculate_cost_function, *args) 

    # This is a linear approximation. Obviously much more efficient
    # if cost function is linear
    if linear_approx:
        f1 = np.matrix(f1)
        return f1.transpose() * f1    
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = np.zeros ( ( n, n ) )
    # The next loop fill in the matrix
    xx = x0
    for j in xrange( n ):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = approx_fprime( x0, calculate_cost_function, *args) 
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0        
    return hessian



####################################

#    def __init__(self, num_layers=3, num_neurons=64, dropout_rate=0.2, activation='relu'):
#        super(CustomModel, self).__init__()
#        self.num_layers = num_layers
#        self.num_neurons = num_neurons
#        self.dropout_rate = dropout_rate
#        self.activation = activation
#
#        self.flatten = tf.keras.layers.Flatten()
#        self.hidden_layers = [tf.keras.layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)]
#        self.dropout = tf.keras.layers.Dropout(dropout_rate)
#        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')'
#
#    def call(self, inputs):
#        x = self.flatten(inputs)
#        for layer in self.hidden_layers:
#            x = layer(x)
#            x = self.dropout(x)
#        return self.output_layer(x) 
#
#################################



