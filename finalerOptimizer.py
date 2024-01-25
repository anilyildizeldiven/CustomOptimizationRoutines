from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export
import tensorflow as tf
import numpy as np

class TestGD(optimizer.Optimizer):
  def __init__(self,
               use_locking=False, name="TestGD"):
    super(TestGD, self).__init__(use_locking, name)

  def _apply_dense(self, grad, var):
    # Flatten the gradient to a 1D vector
    grad_flat = tf.reshape(grad, [-1])
    
    hessian_list = []
    for i in range(len(var)):
        # Berechnung der zweiten Ableitung für den i-ten Parameter
        second_derivative = tf.gradients(grad_flat[i], var)[0]
        hessian_list.append(tf.reshape(second_derivative, [-1]))
    # Zusammenbau der Hesse-Matrix
    hessian_flat = tf.stack(hessian_list, axis=1)
    hessian_square = tf.reshape(hessian_flat, [len(var), len(var)])

    try:
        hessian_inv = tf.linalg.inv(hessian_square)
    except tf.errors.InvalidArgumentError:
        # Fallback, falls die Hesse-Matrix nicht invertierbar ist
        hessian_inv = tf.eye(len(var))

    # Newton-Raphson-Update: -H^(-1) * grad
    update_step = -tf.matmul(hessian_inv, tf.expand_dims(grad_flat, -1))
    
    # Update using TensorFlow operations
    # tf.reshape(update_step, var.shape) statt nur update_step??
    var_update = self._resource_apply_dense(tf.reshape(update_step, var.shape), var)
    return tf.group(var_update)

  def _resource_apply_dense(self, grad, var):
     print(grad.shape, "<-----------")
     var_update = tf.compat.v1.assign_sub(var, grad)
     return tf.group(var_update)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")

