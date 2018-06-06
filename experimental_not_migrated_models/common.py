import argparse
import numpy as np
import tensorflow as tf


# def load_original_mnist():
#     (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
#
#     train_x = prepare_images(train_x)
#     train_y = np.asarray(train_y, dtype=np.int32)
#
#     test_x = prepare_images(test_x)
#     test_y = np.asarray(test_y, dtype=np.int32)
#
#     return (train_x, train_y), (test_x, test_y)
#
#
# def load_original_cifar10():
#     (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
#
#     train_x = prepare_images(train_x)
#     train_y = np.asarray(train_y, dtype=np.int32)
#
#     test_x = prepare_images(test_x)
#     test_y = np.asarray(test_y, dtype=np.int32)
#
#     return (train_x, train_y), (test_x, test_y)
#
#
# def mixed_layer(input_layer, name="mixed_layer"):
#     with tf.name_scope(name):
#         with tf.name_scope("branch_1x1"):
#             branch_1x1 = tf.layers.conv2d(
#                 inputs=input_layer,
#                 filters=32,
#                 kernel_size=[1, 1],
#                 strides=1,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv1x1"
#             )
#
#         with tf.name_scope("branch_3x3"):
#             conv1x1_3x3 = tf.layers.conv2d(
#                 inputs=input_layer,
#                 filters=48,
#                 kernel_size=[1, 1],
#                 strides=1,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv1x1"
#             )
#             branch_3x3 = tf.layers.conv2d(
#                 inputs=conv1x1_3x3,
#                 filters=64,
#                 kernel_size=[3, 3],
#                 strides=1,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv3x3"
#             )
#
#         with tf.name_scope("branch_5x5"):
#             conv1x1_5x5 = tf.layers.conv2d(
#                 inputs=input_layer,
#                 filters=8,
#                 kernel_size=[1, 1],
#                 strides=1,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv1x1"
#             )
#             branch_5x5 = tf.layers.conv2d(
#                 inputs=conv1x1_5x5,
#                 filters=16,
#                 kernel_size=[5, 5],
#                 strides=1,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv5x5"
#             )
#
#         with tf.name_scope("branch_max_pool"):
#             pool3x3 = tf.layers.max_pooling2d(
#                 inputs=input_layer,
#                 pool_size=[3, 3],
#                 strides=1,
#                 padding="same",
#                 name="pool3x3"
#             )
#             branch_max_pool = tf.layers.conv2d(
#                 inputs=pool3x3,
#                 filters=16,
#                 kernel_size=[1, 1],
#                 strides=1,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv1x1"
#             )
#
#         with tf.name_scope("concat_module"):
#             result_layer = tf.concat(
#                 axis=3, values=[branch_1x1, branch_3x3, branch_5x5, branch_max_pool])
#
#     return result_layer
