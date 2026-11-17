import tensorflow as tf
import numpy as np

# Use TensorFlow 1.x style APIs under TensorFlow 2.x
tf.compat.v1.disable_eager_execution()


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(
        shape,
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)