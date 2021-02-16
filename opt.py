import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
import keras.backend as K
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp
from tensorflow.python.ops import state_ops, control_flow_ops, math_ops, array_ops


#
# Adapted from
#       https://www.tensorflow.org/tutorials/structured_data/feature_columns
#       https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
#       https://keras.io/api/optimizers/
#


class WAME(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=0.01, name='WAME', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('eta_plus', tf.Variable(1.2))
        self._set_hyper('eta_minus', tf.Variable(0.1))
        self._set_hyper('eta_max', tf.Variable(100.0))
        self._set_hyper('eta_min', tf.Variable(0.01))
        self._set_hyper('rho', tf.Variable(0.9))
        self.epsilon = 1e-7

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }

    def _create_slots(self, var_list):
        [self.add_slot(var, 'prev_grad', initializer=tf.ones) for var in var_list]
        [self.add_slot(var, 'beta', initializer=tf.zeros) for var in var_list]
        [self.add_slot(var, 'zeta', initializer=tf.ones) for var in var_list]
        [self.add_slot(var, 'zed', initializer=tf.zeros) for var in var_list]

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(WAME, self)._prepare_local(var_device, var_dtype, apply_state)
        rho = array_ops.identity(self._get_hyper('rho', var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                rho=rho,
                one_minus_rho=1. - rho))

    def _resource_apply_dense(self, grad, var, apply_state):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        prev_grad = self.get_slot(var, 'prev_grad')
        beta = self.get_slot(var, 'beta')
        zeta = self.get_slot(var, 'zeta')
        zedd = self.get_slot(var, 'zed')
        gt_g = grad * prev_grad
        zeta_t = tf.where(tf.greater(gt_g, tf.zeros_like(gt_g)),
                          tf.minimum(zeta * self._get_hyper('eta_plus'), self._get_hyper('eta_max')),
                          tf.maximum(zeta * self._get_hyper('eta_minus'), self._get_hyper('eta_min')))
        zedd_t = coefficients['rho'] * zedd + coefficients['one_minus_rho'] * zeta_t
        beta_t = coefficients['rho'] * beta + coefficients['one_minus_rho'] * grad**2
        var_t = var - (coefficients['lr_t'] / zedd_t) * grad / (math_ops.sqrt(beta_t) + coefficients['epsilon'])
        beta.assign(beta_t, use_locking=self._use_locking)
        prev_grad.assign(grad, use_locking=self._use_locking)
        zeta.assign(zeta_t, use_locking=self._use_locking)
        zedd.assign(zedd_t, use_locking=self._use_locking)
        return state_ops.assign(var, var_t, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        raise NotImplementedError
