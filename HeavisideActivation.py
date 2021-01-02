import tensorflow as tf


@tf.custom_gradient
def heaviside(x):
    sign = tf.sign(x)
    # tf.stop_gradient is needed to exclude tf.maximum from derivative
    z = tf.stop_gradient(tf.maximum(0.0, sign))

    def g(grad):
        return 1 * grad

    return z, g