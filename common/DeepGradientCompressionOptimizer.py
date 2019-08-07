import tensorflow as tf
import re
import numpy as np

class DeepGradientCompressionOptimizer(tf.train.Optimizer):
   def __init__(self, optimizer, name=None):
       if name is None:
           name = "DeepGradientCompressionOptimizer{}".format(type(optimizer).__name__)

       self._optimizer = optimizer
       self._name = name

   def compute_gradients(self, *args, **kwargs):
       grads_and_vars = self._optimizer.compute_gradients(*args, **kwargs)
       grads_and_vars = [(g,v) for g,v in grads_and_vars if g is not None]

       # IndexSparsification ends over here
       indexedSliced_grads = []
       threshold = tf.contrib.distributions.percentile(tf.abs(grads_and_vars[0][0]),98.0,interpolation='higher')
       for grad, var in grads_and_vars:
           #threshold = tf.contrib.distributions.percentile(tf.abs(grad),90.0,interpolation='higher')
           prev_grad = self._optimizer._get_or_make_slot(var, tf.zeros(tf.shape(grad), grad.dtype, 'prev_grad'), 'prev_grad', self._name)
           grad = tf.math.add(grad, prev_grad)

           # backed up grad that are less than threshold to use in next iteration
           bool_mask_less = tf.math.less(abs(grad), threshold)
           float_mask_less = tf.cast(bool_mask_less, grad.dtype)
           backup_grads = tf.multiply(grad, float_mask_less)
           prev_grad = self._optimizer._get_or_make_slot(var, backup_grads, 'prev_grad', self._name)

           #create an Indexed slices method
           flat_grad = tf.reshape(grad, [-1])
           bool_mask = tf.math.greater(tf.abs(flat_grad), threshold)
           indices = tf.reshape(tf.where(bool_mask),[-1])
           values = tf.reshape(tf.gather(flat_grad, indices),[-1])

           indexed_sclices = tf.IndexedSlices(values,indices,dense_shape=grad.shape)
           indexedSliced_grads.append(indexed_sclices)
           # IndexSparsification ends over here

       return [(grad, gradvar[1]) for grad, gradvar in zip(indexedSliced_grads, grads_and_vars)]


   def apply_gradients(self, *args, **kwargs):
       return self._optimizer.apply_gradients(*args, **kwargs)

   def get_slot(self, *args, **kwargs):
       return self._optimizer.get_slot(*args, **kwargs)

   def get_slot_names(self, *args, **kwargs):
       return self._optimizer.get_slot_names(*args, **kwargs)

   def variables(self, *args, **kwargs):
       return self._optimizer.variables(*args, **kwargs)

   def _create_slots(self, var_list):
       for v in var_list:
           self._zeros_slot(v, "prev_grad", self._name)

   def sparse_to_dense(self, grads_and_vars):
       # convert Indexed Slices returned by horovod to dense gradients
       dense_grads = []
       for sparse_grad, var in grads_and_vars:
           if isinstance(sparse_grad, tf.IndexedSlices):
               indices = tf.cast(sparse_grad.indices,tf.int32)
               values = sparse_grad.values
               shape = sparse_grad.dense_shape
               dimensions,multiple = [],1
               if shape is not None:
                   dimensions = shape.as_list()
               if dimensions is not None:
                   if len(dimensions) == 1:
                       multiple = dimensions[0]
                   else:
                       for dimension in dimensions:
                           multiple = multiple * dimension
           grad = tf.reshape(tf.sparse_to_dense(indices,[multiple],values, default_value=0, validate_indices=True, name=None), shape)
           dense_grads.append(grad)
       grads_and_vars = [(grad, gradvar[1]) for grad, gradvar in zip(dense_grads, grads_and_vars)]
       return grads_and_vars
