import tensorflow as tf


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(var.name.replace(":", "_")):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def build_layer_summaries(layer_name):
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=layer_name):
        variable_summaries(var)


def clean_dir(dir_name):
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)
