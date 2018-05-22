from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pprint
import numpy as np
import tensorflow as tf

import common
from image import LabeledImage
from image_dataset import ImageDataset
import image_dataset as ds


def build_app_flags():
    # Hyperparameters for the model training/evaluation
    # They are accessible everywhere in the application via tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string("model_dir", "../models/stl10/test16",
                               "Model checkpoint/training/evaluation data directory")
    tf.app.flags.DEFINE_float("dropout_rate", 0.4,
                              "Dropout rate for model training")
    tf.app.flags.DEFINE_integer("eval_batch_size", 5,  # 256
                                "Evaluation data batch size")
    tf.app.flags.DEFINE_integer("train_batch_size", 2,  # 128
                                "Training data batch size")
    tf.app.flags.DEFINE_float("initial_learning_rate", 0.05,
                              "Initial value for learning rate")
    tf.app.flags.DEFINE_bool("use_static_learning_rate", False,
                             "Flag that determines if learning rate should be constant value")
    tf.app.flags.DEFINE_float("learning_rate_decay_rate", 0.96,
                              "Learning rate decay rate")
    tf.app.flags.DEFINE_integer("learning_rate_decay_steps", 5000,
                                "Learning rate decay steps")
    tf.app.flags.DEFINE_bool("ignore_gpu", False,
                             "Flag that determines if gpu should be disabled")
    tf.app.flags.DEFINE_float("per_process_gpu_memory_fraction", 1.0,
                              "Fraction of gpu memory to be used")


def get_model_params():
    return {"add_layer_summaries": True}


def load_train_dataset():
    dataset = ImageDataset.load_from_pickles([
        "/datasets/stl10/original_train.pkl",
        "/datasets/stl10/mirror_train.pkl",
        "/datasets/stl10/rand_distorted_train.pkl",
        # "/datasets/stl10/rand_distorted_train_0.pkl",
        # "/datasets/stl10/rand_distorted_train_1.pkl",
        # "/datasets/stl10/rand_distorted_train_2.pkl",
    ])

    return dataset.x, dataset.y


def load_eval_dataset():
    dataset = ImageDataset.load_from_pickles([
        "/datasets/stl10/original_test.pkl",
    ])

    return dataset.x, dataset.y


def load_original(images_dtype=np.float16, labels_dtype=np.uint8):
    common.maybe_download_and_extract(
        dest_dir="../data",
        data_url="http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
        nested_dir="stl10_binary"
    )

    # data paths
    train_x_path = '../data/stl10_binary/train_X.bin'
    train_y_path = '../data/stl10_binary/train_y.bin'

    test_x_path = '../data/stl10_binary/test_X.bin'
    test_y_path = '../data/stl10_binary/test_y.bin'

    def read_labels(path_to_labels):
        with open(path_to_labels, 'rb') as f:
            return np.fromfile(f, dtype=np.uint8)

    def read_images(path_to_data):
        with open(path_to_data, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))

            return np.transpose(images, (0, 3, 2, 1))

    # load images/labels from binary file
    train_x = read_images(train_x_path)
    train_y = read_labels(train_y_path)

    test_x = read_images(test_x_path)
    test_y = read_labels(test_y_path)

    # prepare images/labels for training
    train_x = common.prepare_images(train_x, dtype=images_dtype)
    train_y = np.asarray(train_y, dtype=labels_dtype)

    test_x = common.prepare_images(test_x, dtype=images_dtype)
    test_y = np.asarray(test_y, dtype=labels_dtype)

    return (train_x, np.add(train_y, -1)), (test_x, np.add(test_y, -1))


def model_fn(features, labels, mode, params, config):
    """Model function for CNN."""

    app_flags = tf.app.flags.FLAGS

    print("Model directory: " + config.model_dir)

    # Add name to labels tensor
    labels = tf.identity(labels, name="labels")

    # Input Layer
    with tf.name_scope("input_layer"):
        input_layer = features["x"]

    # Convolution 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[7, 7],
        strides=2,
        padding="same",
        activation=tf.nn.relu,
        name="conv1"
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[3, 3],
        strides=2,
        padding="same",
        name="pool1"
    )

    # Convolution 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        strides=1,
        padding="same",
        activation=tf.nn.relu,
        name="conv2"
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2,
        name="pool2"
    )

    # Convolution 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=172,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu,
        name="conv3"
    )
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=2,
        name="pool3"
    )

    # Convolution 4
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu,
        name="conv4"
    )

    # Flatten output of the last convolution
    pool_flat = tf.reshape(
        conv4, [-1, conv4.shape[1]*conv4.shape[2]*conv4.shape[3]], name="pool_flat")

    # Dense Layers
    dense1 = tf.layers.dense(
        inputs=pool_flat,
        units=1024,
        activation=tf.nn.relu,
        name="dense1"
    )

    dense2 = tf.layers.dense(
        inputs=dense1,
        units=512,
        activation=tf.nn.relu,
        name="dense2"
    )

    dense3 = tf.layers.dense(
        inputs=dense2,
        units=256,
        activation=tf.nn.relu,
        name="dense3"
    )

    if params.get("add_layer_summaries", False) is True:
        weighted_layers_names = ["conv1", "conv2", "conv3", "conv4", "dense1", "dense2", "dense3"]
        for layer_name in weighted_layers_names:
            common.build_layer_summaries(layer_name)

    with tf.name_scope("dropout"):
        dropout = tf.layers.dropout(
            inputs=dense3, rate=app_flags.dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10, name="logits")

    argmax = tf.argmax(input=logits, axis=1, name="predictions", output_type=tf.int32)

    top_k_values, top_k_indices = tf.nn.top_k(input=logits, k=2)
    tf.identity(top_k_values, "top_k_values")
    tf.identity(top_k_indices, "top_k_indices")

    softmax = tf.nn.softmax(logits, name="softmax_tensor")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": argmax,
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": softmax
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, scope="calc_loss")
    tf.summary.scalar("cross_entropy", loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = common.get_learning_rate_from_flags(tf.app.flags.FLAGS)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate, name="gradient_descent_optimizer")
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="adam_optimizer")
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step(), name="minimize_loss")

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops,
    )


def split_dataset(ds, classes_count=10, test_items_per_class=300):
    count_per_class = np.zeros(classes_count, int)
    new_ds_x = []
    new_ds_y = []
    indexes = []

    ds_x, ds_y = ds

    for idx, label in enumerate(ds_y):
        if count_per_class[label] < test_items_per_class:
            new_ds_x.append(ds_x[idx])
            new_ds_y.append(label)
            count_per_class[label] += 1
            indexes.append(idx)

    ds_x = np.delete(ds_x, indexes, axis=0)
    ds_y = np.delete(ds_y, indexes, axis=0)
    test_ds = (np.asarray(new_ds_x), np.asarray(new_ds_y))
    train_ds = (ds_x, ds_y)

    return train_ds, test_ds


if __name__ == "__main__":
    train, test = load_original()

    extra_train, test = split_dataset(test)

    train_x = np.concatenate([train[0], extra_train[0]], axis=0)
    train_y = np.concatenate([train[1], extra_train[1]], axis=0)
    train = (train_x, train_y)

    print(test[0].shape, test[0].dtype, test[1].shape, test[1].dtype)
    print(train[0].shape, train[0].dtype, train[1].shape, train[1].dtype)

    print(sys.getsizeof(train[0]) // (1024*1024), sys.getsizeof(train[1]) // (1024*1024))

    ds.improve_dataset(
        train,
        test,
        "stl10",
        crop_shape=(72, 72, 3),
        target_size=96,
        rand_dist_sets=3,
        save_location="../datasets"
    )

