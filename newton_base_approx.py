#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:43:38 2023

@author: anilcaneldiven
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.framework import tensor
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_training_ops

class NewtonOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=0.01, name="NewtonOptimizer", **kwargs):
        super(NewtonOptimizer, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "accumulator")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        accumulator = self.get_slot(var, "accumulator")
        new_accumulator = accumulator + grad * grad  # Update accumulator with square of gradients

        var_update = var - coefficients["lr_t"] * grad / (new_accumulator + 1e-8)  # Update var using Newton's method

        var.assign(var_update)  # Assign the updated value back to the variable

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        accumulator = self.get_slot(var, "accumulator")
        new_accumulator = accumulator + grad * grad  # Update accumulator with square of gradients

        var_update = var - coefficients["lr_t"] * grad / (new_accumulator + 1e-8)  # Update var using Newton's method

        var.scatter_sub(indices, var - var_update)  # Update var for sparse tensors

    def get_config(self):
        config = super(NewtonOptimizer, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate")
        })
        return config


