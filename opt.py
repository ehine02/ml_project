import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
import keras.backend as K
from tensorflow.python.ops import state_ops

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
        self._set_hyper('alpha', tf.Variable(0.9))

    def _create_slots(self, var_list):
        [self.add_slot(var, 'zeta', initializer=tf.ones) for var in var_list]
        [self.add_slot(var, 'beta', initializer=tf.zeros) for var in var_list]
        [self.add_slot(var, 'z', initializer=tf.zeros) for var in var_list]
        [self.add_slot(var, 'prev_grad', initializer=tf.ones) for var in var_list]

    def _resource_apply_dense(self, grad, var, apply_state):
        prev_grad = self.get_slot(var, 'prev_grad')
        prev_zeta = self.get_slot(var, 'zeta')
        prev_z = self.get_slot(var, 'z')
        prev_beta = self.get_slot(var, 'beta')

        def grad_min():
            return tf.minimum(prev_zeta * self._get_hyper('eta_plus'), self._get_hyper('eta_max'))

        def grad_max():
            return tf.maximum(prev_zeta * self._get_hyper('eta_minus'), self._get_hyper('eta_min'))

        zeta_t = tf.case([(tf.greater(grad * prev_grad, 0), grad_min),
                          (tf.less(grad * prev_grad, 0), grad_max)],
                         exclusive=False)
        z_t = self._get_hyper('alpha') * prev_z + (1 - self._get_hyper('alpha')) * zeta_t
        beta_t = self._get_hyper('alpha') * prev_beta + (1 - self._get_hyper('alpha')) * grad ** 2
        dw_t = (-1 * self._get_hyper('learning_rate') / z_t) * grad * (1 / beta_t)
        var_t = var + dw_t
        var.assign(var_t)
        prev_grad.assign(grad)
        prev_zeta.assign(zeta_t)
        prev_z.assign(z_t)
        prev_beta.assign(beta_t)
        return state_ops.assign(var, var_t, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        raise NotImplementedError
