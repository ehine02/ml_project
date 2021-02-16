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


class WAMEOptimizer(optimizer_v2.OptimizerV2):
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }

    def __init__(self, learning_rate=0.01, name='WAMEOptimizer', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('eta_plus', tf.Variable(1.2))
        self._set_hyper('eta_minus', tf.Variable(0.1))
        self._set_hyper('eta_max', tf.Variable(100.0))
        self._set_hyper('eta_min', tf.Variable(0.01))
        self._set_hyper('rho', tf.Variable(0.9))
        self.epsilon = 1e-7

    def _create_slots(self, var_list):
        [self.add_slot(var, 'zeta', initializer=tf.ones) for var in var_list]
        [self.add_slot(var, 'beta', initializer=tf.zeros) for var in var_list]
        [self.add_slot(var, 'zed', initializer=tf.zeros) for var in var_list]
        [self.add_slot(var, 'prev_grad', initializer=tf.ones) for var in var_list]
        [self.add_slot(var, 'rms') for var in var_list]

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(WAMEOptimizer, self)._prepare_local(var_device, var_dtype, apply_state)

        rho = array_ops.identity(self._get_hyper('rho', var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                #neg_lr_t=-apply_state[(var_device, var_dtype)]['lr_t'],
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                rho=rho,
                one_minus_rho=1. - rho))

    def _resource_apply_dense(self, grad, var, apply_state):
        # prev_grad = self.get_slot(var, 'prev_grad')
        # prev_zeta = self.get_slot(var, 'zeta')
        # prev_zed = self.get_slot(var, 'zed')
        # prev_beta = self.get_slot(var, 'beta')
        #
        # def grad_min():
        #     return tf.minimum(prev_zeta * self._get_hyper('eta_plus'), self._get_hyper('eta_max'))
        #
        # def grad_max():
        #     return tf.maximum(prev_zeta * self._get_hyper('eta_minus'), self._get_hyper('eta_min'))
        #
        # zeta_t = tf.case([(tf.greater(grad * prev_grad, 0), grad_min),
        #                   (tf.less(grad * prev_grad, 0), grad_max)],
        #                  exclusive=False)
        # zed_t = self._get_hyper('rho') * prev_zed + (1 - self._get_hyper('rho')) * zeta_t
        # beta_t = self._get_hyper('rho') * prev_beta + (1 - self._get_hyper('rho')) * grad ** 2
        # dw_t = (-1 * self._get_hyper('learning_rate') / zed_t) * grad * (1 / beta_t)
        # var_t = var + dw_t
        # #var.assign(var_t)
        # g_t = prev_grad.assign(grad, use_locking=self._use_locking)
        # zt_t = prev_zeta.assign(zeta_t, use_locking=self._use_locking)
        # z_t = prev_zed.assign(zed_t, use_locking=self._use_locking)
        # b_t = prev_beta.assign(beta_t, use_locking=self._use_locking)
        # var_update = var.assign_sub(var_t, use_locking=self._use_locking)
        # updates = [var_update, g_t, zt_t, z_t, b_t]
        # return tf.group(*updates)
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        prev_beta = self.get_slot(var, 'beta')
        prev_grad = self.get_slot(var, 'prev_grad')
        prev_zeta = self.get_slot(var, 'zeta')
        prev_zed = self.get_slot(var, 'zed')

        g_x_gt = grad * prev_grad

        zeta_t = tf.where(tf.greater(g_x_gt, tf.zeros_like(g_x_gt)),
                          tf.minimum(prev_zeta * self._get_hyper('eta_plus'), self._get_hyper('eta_max')),
                          tf.maximum(prev_zeta * self._get_hyper('eta_minus'), self._get_hyper('eta_min')))
        zed_t = coefficients['rho'] * prev_zed + coefficients['one_minus_rho'] * zeta_t
        #rms = self.get_slot(var, 'rms')
        #rms_t = (coefficients['rho'] * rms + coefficients['one_minus_rho'] * math_ops.square(grad))
        beta_t = coefficients['rho'] * prev_beta + coefficients['one_minus_rho'] * grad**2
        #rms_t = state_ops.assign(rms, rms_t, use_locking=self._use_locking)
        #denom_t = rms_t
        var_t = var - (coefficients['lr_t'] / zed_t) * grad / (math_ops.sqrt(beta_t) + coefficients['epsilon'])
        prev_beta.assign(beta_t, use_locking=self._use_locking)
        prev_grad.assign(grad, use_locking=self._use_locking)
        prev_zeta.assign(zeta_t, use_locking=self._use_locking)
        prev_zed.assign(zed_t, use_locking=self._use_locking)
        return state_ops.assign(var, var_t, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        raise NotImplementedError


class WAMEFromRMSProp(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=0.01, name='WAMEFromRMSProp', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('eta_plus', tf.Variable(1.2))
        self._set_hyper('eta_minus', tf.Variable(0.1))
        self._set_hyper('eta_max', tf.Variable(100.0))
        self._set_hyper('eta_min', tf.Variable(0.01))
        self._set_hyper('rho', tf.Variable(0.9))
        self.rms_prop_optimizer = RMSProp(learning_rate)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
        }

    def _create_slots(self, var_list):
        [self.add_slot(var, 'zeta', initializer=tf.ones) for var in var_list]
        [self.add_slot(var, 'z', initializer=tf.zeros) for var in var_list]
        [self.add_slot(var, 'prev_grad', initializer=tf.ones) for var in var_list]
        self.rms_prop_optimizer._create_slots(var_list)
        self.apply_state = self.rms_prop_optimizer._prepare(var_list)

    def _resource_apply_dense(self, grad, var, apply_state):
        prev_grad = self.get_slot(var, 'prev_grad')
        prev_zeta = self.get_slot(var, 'zeta')
        prev_z = self.get_slot(var, 'z')

        def grad_min():
            return tf.minimum(prev_zeta * self._get_hyper('eta_plus'), self._get_hyper('eta_max'))

        def grad_max():
            return tf.maximum(prev_zeta * self._get_hyper('eta_minus'), self._get_hyper('eta_min'))

        zeta_t = tf.case([(tf.greater(grad * prev_grad, 0), grad_min),
                          (tf.less(grad * prev_grad, 0), grad_max)],
                         exclusive=False)
        z_t = self._get_hyper('rho') * prev_z + (1 - self._get_hyper('rho')) * zeta_t
        self.rms_prop_optimizer._resource_apply_dense(grad / z_t, var, self.apply_state)
