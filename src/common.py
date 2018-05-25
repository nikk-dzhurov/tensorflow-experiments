import os
import sys
import six
import urllib
import tarfile
import argparse
import numpy as np
import tensorflow as tf

from image import LabeledImage

EVAL_MODE = "eval"
TRAIN_MODE = "train"
PREDICT_MODE = "predict"
TRAIN_EVAL_MODE = "train_eval"


def parse_known_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--clean',
        type=bool,
        default=False,
        help="Remove all model data and start new training"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=TRAIN_EVAL_MODE,
        help="Model mode"
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=0,
        help="Set training epochs"
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=0,
        help="Set training steps per epoch"
    )

    parsed_args, _ = parser.parse_known_args()

    return parsed_args


def duration_to_string(dur_in_sec=0):
    days, remainder = divmod(dur_in_sec, 60*60*24)
    hours, remainder = divmod(remainder, 60*60)
    minutes, seconds = divmod(remainder, 60)
    output = ""
    if days > 0:
        output += "%d days, " % days
    if hours > 0:
        output = "%d hours, " % hours
    if minutes > 0:
        output += "%d min, " % minutes
    if seconds > 0 or len(output) == 0:
        output += "%.3f sec" % seconds
    if output[-2:] == ", ":
        output = output[:-2]

    return output


def prepare_images(images, dtype=np.float32):
    images = np.asarray(images, dtype=dtype)

    return np.multiply(images, 1.0 / 255.0)


def load_original_mnist():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = prepare_images(train_x)
    train_y = np.asarray(train_y, dtype=np.int32)

    test_x = prepare_images(test_x)
    test_y = np.asarray(test_y, dtype=np.int32)

    return (train_x, train_y), (test_x, test_y)


def load_original_cifar10():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

    train_x = prepare_images(train_x)
    train_y = np.asarray(train_y, dtype=np.int32)

    test_x = prepare_images(test_x)
    test_y = np.asarray(test_y, dtype=np.int32)

    return (train_x, train_y), (test_x, test_y)


def get_learning_rate_from_flags(flags):
    if flags.use_static_learning_rate:
        learning_rate = flags.initial_learning_rate
    else:
        learning_rate = tf.train.exponential_decay(
            learning_rate=flags.initial_learning_rate,
            global_step=tf.train.get_global_step(),
            decay_steps=flags.learning_rate_decay_steps,
            decay_rate=flags.learning_rate_decay_rate,
            name="learning_rate"
        )

    return learning_rate


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


def mixed_layer(input_layer, name="mixed_layer"):
    with tf.name_scope(name):
        with tf.name_scope("branch_1x1"):
            branch_1x1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[1, 1],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv1x1"
            )

        with tf.name_scope("branch_3x3"):
            conv1x1_3x3 = tf.layers.conv2d(
                inputs=input_layer,
                filters=48,
                kernel_size=[1, 1],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv1x1"
            )
            branch_3x3 = tf.layers.conv2d(
                inputs=conv1x1_3x3,
                filters=64,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv3x3"
            )

        with tf.name_scope("branch_5x5"):
            conv1x1_5x5 = tf.layers.conv2d(
                inputs=input_layer,
                filters=8,
                kernel_size=[1, 1],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv1x1"
            )
            branch_5x5 = tf.layers.conv2d(
                inputs=conv1x1_5x5,
                filters=16,
                kernel_size=[5, 5],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv5x5"
            )

        with tf.name_scope("branch_max_pool"):
            pool3x3 = tf.layers.max_pooling2d(
                inputs=input_layer,
                pool_size=[3, 3],
                strides=1,
                padding="same",
                name="pool3x3"
            )
            branch_max_pool = tf.layers.conv2d(
                inputs=pool3x3,
                filters=16,
                kernel_size=[1, 1],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name="conv1x1"
            )

        with tf.name_scope("concat_module"):
            result_layer = tf.concat(
                axis=3, values=[branch_1x1, branch_3x3, branch_5x5, branch_max_pool])

    return result_layer
