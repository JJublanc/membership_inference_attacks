from absl import logging
import collections

from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query

from tensorflow.keras.callbacks import *

import tensorflow as tf

from tensorflow.python.keras import backend as K

from tensorflow.python import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables

import numpy as np

def make_optimizer_class(cls):
  """Constructs a DP optimizer class from an existing one."""
  parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
  child_code = cls.compute_gradients.__code__
  GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
  if child_code is not parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):
    """Differentially private subclass of given class cls."""

    _GlobalState = collections.namedtuple(
      '_GlobalState', ['l2_norm_clip', 'stddev'])
    
    def __init__(
        self,
        dp_sum_query,
        num_microbatches=None,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        dp_sum_query: DPQuery object, specifying differential privacy
          mechanism to use.
        num_microbatches: How many microbatches into which the minibatch is
          split. If None, will default to the size of the minibatch, and
          per-example gradients will be computed.
        unroll_microbatches: If true, processes microbatches within a Python
          loop instead of a tf.while_loop. Can be used if using a tf.while_loop
          raises an exception.
      """
      super(DPOptimizerClass, self).__init__(*args, **kwargs)
      self._dp_sum_query = dp_sum_query
      self._num_microbatches = num_microbatches
      self._global_state = self._dp_sum_query.initial_global_state()
      # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
      # Beware: When num_microbatches is large (>100), enabling this parameter
      # may cause an OOM error.
      self._unroll_microbatches = unroll_microbatches

    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None,
                          gradient_tape=None,
                          curr_noise_mult=0,
                          curr_norm_clip=1):
      
      self._dp_sum_query = gaussian_query.GaussianSumQuery(curr_norm_clip, 
                                                           curr_norm_clip*curr_noise_mult)
      self._global_state = self._dp_sum_query.make_global_state(curr_norm_clip, 
                                                                curr_norm_clip*curr_noise_mult)
      

      # TF is running in Eager mode, check we received a vanilla tape.
      if not gradient_tape:
        raise ValueError('When in Eager mode, a tape needs to be passed.')

      vector_loss = loss()
      if self._num_microbatches is None:
        self._num_microbatches = tf.shape(input=vector_loss)[0]
      sample_state = self._dp_sum_query.initial_sample_state(var_list)
      microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
      sample_params = (self._dp_sum_query.derive_sample_params(self._global_state))

      def process_microbatch(i, sample_state):
        """Process one microbatch (record) with privacy helper."""
        microbatch_loss = tf.reduce_mean(input_tensor=tf.gather(microbatches_losses, [i]))
        grads = gradient_tape.gradient(microbatch_loss, var_list)
        sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)
        return sample_state
    
      for idx in range(self._num_microbatches):
        sample_state = process_microbatch(idx, sample_state)

      if curr_noise_mult > 0:
        grad_sums, self._global_state = (self._dp_sum_query.get_noised_result(sample_state, self._global_state))
      else:
        grad_sums = sample_state

      def normalize(v):
        return v / tf.cast(self._num_microbatches, tf.float32)

      final_grads = tf.nest.map_structure(normalize, grad_sums)
      grads_and_vars = final_grads#list(zip(final_grads, var_list))
    
      return grads_and_vars

  return DPOptimizerClass


def make_gaussian_optimizer_class(cls):
  """Constructs a DP optimizer with Gaussian averaging of updates."""

  class DPGaussianOptimizerClass(make_optimizer_class(cls)):
    """DP subclass of given class cls using Gaussian averaging."""

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        ledger=None,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):
      dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)
      print("noise multiplier : ", noise_multiplier, "l2 norm clip : ", l2_norm_clip)
      if ledger:
        dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query,
                                                      ledger=ledger)

      super(DPGaussianOptimizerClass, self).__init__(
          dp_sum_query,
          num_microbatches,
          unroll_microbatches,
          *args,
          **kwargs)

    @property
    def ledger(self):
      return self._dp_sum_query.ledger

  return DPGaussianOptimizerClass


class LearningRateScheduler_Perso(Callback):
  """Learning rate scheduler.
  Arguments:
      schedule: a function that takes an epoch index as input
          (integer, indexed from 0) and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages.
  ```python
  # This function keeps the learning rate at 0.001 for the first ten epochs
  # and decreases it exponentially after that.
  def scheduler(epoch):
    if epoch < 10:
      return 0.001
    else:
      return 0.001 * tf.math.exp(0.1 * (10 - epoch))
  callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
  model.fit(data, labels, epochs=100, callbacks=[callback],
            validation_data=(val_data, val_labels))
  ```
  """

  def __init__(self, schedule, verbose=0):
    super(LearningRateScheduler_Perso, self).__init__()
    self.schedule = schedule
    self.verbose = verbose
    
  def on_epoch_begin(self, epoch, logs=None):
    if hasattr(self.model.optimizer, 'lr'):
        try:  # new API
          lr = float(K.get_value(self.model.optimizer.lr))
          lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
          lr = self.schedule(epoch)
    elif hasattr(self.model.optimizer.optimizer, '_learning_rate'):
        try:  # new API
          lr = float(K.get_value(self.model.optimizer.optimizer._learning_rate))
          lr = self.schedule(epoch, lr)
          print(lr)
        except TypeError:  # Support for old API for backward compatibility
          lr = self.schedule(epoch)
    else:
        raise ValueError('Optimizer must have a "lr" or "_learning_rate" attribute.')
        
    if not isinstance(lr, (ops.Tensor, float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
      raise ValueError('The dtype of Tensor should be float')
    
    if hasattr(self.model.optimizer, 'lr'):
        K.set_value(self.model.optimizer.lr, K.get_value(lr))
        if self.verbose > 0:
          print('\nEpoch %05d: LearningRateScheduler reducing learning '
                'rate to %s.' % (epoch + 1, lr))
    elif hasattr(self.model.optimizer.optimizer, '_learning_rate'):
        #K.set_value(self.model.optimizer.optimizer._learning_rate, tf.convert_to_tensor(lr))
        self.model.optimizer.optimizer._learning_rate=lr
        if self.verbose > 0:
          print('\nEpoch %05d: LearningRateScheduler reducing learning '
                'rate to %s.' % (epoch + 1, lr))
        
  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    if hasattr(self.model.optimizer, 'lr'):
        logs['lr'] = K.get_value(self.model.optimizer.lr)
    elif hasattr(self.model.optimizer.optimizer, '_learning_rate'):
        logs['lr'] = K.get_value(self.model.optimizer.optimizer._learning_rate)