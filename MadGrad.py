"""Madgrad optimizer implementation."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops, control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class MadGrad(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            learning_rate=1e-2,
            momentum=0.9,
            weight_decay=0.0,
            power=1.0 / 3.0,
            epsilon=1e-6,
            name="Madgrad",
            **kwargs
    ):
        super(MadGrad, self).__init__(name, **kwargs)
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if learning_rate <= 0:
            raise ValueError(f"Learning rate {learning_rate} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if epsilon < 0:
            raise ValueError(f"Eps must be non-negative")
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("momentum", momentum)
        self._set_hyper("power", power)
        self._set_hyper("weight_decay", weight_decay)
        self.epsilon = epsilon or backend_config.epsilon()
        self.apply_weight_decay = weight_decay > 0.0

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "grad_sum_sq")
            self.add_slot(var, "s")
            self.add_slot(var, "x0", initializer=var)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(MadGrad, self)._prepare_local(var_device, var_dtype, apply_state)
        power = array_ops.identity(self._get_hyper("power", var_dtype))
        momentum = array_ops.identity(self._get_hyper("momentum", var_dtype))
        weight_decay = array_ops.identity(self._get_hyper("weight_decay", var_dtype))
        lr_t = apply_state[(var_device, var_dtype)]['lr_t']
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        apply_state[(var_device, var_dtype)] = dict(
            epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
            power=power,
            momentum=momentum,
            one_minus_momentum_t=1.0 - momentum,
            weight_decay=weight_decay,
            lamb=lr_t * math_ops.sqrt(local_step),
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        grad_sum_sq = self.get_slot(var, "grad_sum_sq")
        s = self.get_slot(var, "s")
        x0 = self.get_slot(var, "x0")
        if self.apply_weight_decay:
            grad += coefficients["weight_decay"] * var
        sk_grad = coefficients["lamb"] * grad
        s_t = state_ops.assign_add(s, sk_grad, use_locking=self._use_locking)
        grad_sum_sq_t = state_ops.assign_add(grad_sum_sq, sk_grad * grad, use_locking=self._use_locking)
        rms = math_ops.maximum(math_ops.pow(grad_sum_sq_t, coefficients["power"]), coefficients["epsilon"])
        z = x0 - (s_t / rms)
        var_t = coefficients['one_minus_momentum_t'] * var + coefficients["momentum"] * z
        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)
        updates = [s_t, grad_sum_sq_t, var_update]
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        grad_sum_sq = self.get_slot(var, "grad_sum_sq")
        s = self.get_slot(var, "s")
        x0 = self.get_slot(var, "x0")
        if self.apply_weight_decay:
            grad += coefficients["weight_decay"] * array_ops.gather(var, indices)
        sk_grad = coefficients["lamb"] * grad
        s_t = self._resource_scatter_add(s, indices, sk_grad)
        grad_sum_sq_t = self._resource_scatter_add(grad_sum_sq, indices, sk_grad * grad)
        rms = math_ops.maximum(math_ops.pow(grad_sum_sq_t, coefficients["power"]), coefficients["epsilon"])
        z = x0 - (s_t / rms)
        var_t = coefficients['one_minus_momentum_t'] * var + coefficients["momentum"] * z
        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)
        updates = [s_t, grad_sum_sq_t, var_update]
        return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(MadGrad, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "power": self._serialize_hyperparameter("power"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "epsilon": self.epsilon,
            }
        )
        return config
