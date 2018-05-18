import os
import sys
import six
import json
import copy
import pickle
import urllib
import tarfile
import argparse
import numpy as np
import tensorflow as tf

from image import LabeledImage

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
        default=tf.estimator.ModeKeys.TRAIN,
        help="Model mode"
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
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


def get_final_eval_result(results=None):
    res = {"error": "result is missing"}
    if results is not None and len(results) > 0:
        res = results[-1].copy()
        res["accuracy"] = res["accuracy"].item()
        res["loss"] = res["loss"].item()
        res["global_step"] = res["global_step"].item()

    return res


def save_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
        print("%s saved" % filename)


def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)


def save_pickle(data, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)
        print("%s saved" % filename)


def load_pickle(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


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


def _track_progress(count, block_size, total_size):
    progress = float(count * block_size) / float(total_size) * 100.0
    sys.stdout.write('\r\t>> Progress %.2f%%' % progress)
    sys.stdout.flush()


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


def maybe_download_and_extract(dest_dir, data_url, nested_dir):
    # Example usage:
    # maybe_download_and_extract(
    #     dest_dir="./test-data",
    #     data_url='https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
    #     nested_dir=os.path.join(os.getcwd(), "test-data/cifar-10-batches-bin")
    # )
    """Download and extract data from tarball"""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)

    if not os.path.exists(filepath):
        print("\nDownloading %s" % filename)
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _track_progress)
        statinfo = os.stat(filepath)
        print('\nSuccessfully downloaded %s : %d bytes.' % (filename, statinfo.st_size))
    else:
        statinfo = os.stat(filepath)
        print('\nFile is already downloaded %s : %d bytes.' % (filename, statinfo.st_size))

    extracted_dir = os.path.join(dest_dir, nested_dir)
    if not os.path.exists(extracted_dir):
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)
        print('\File %s is extracted successfully at %s' % (filename, extracted_dir))
    else:
        print('File %s is already extracted successfully at %s' % (filename, extracted_dir))



