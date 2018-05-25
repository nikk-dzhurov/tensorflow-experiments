from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
import tensorflow as tf

import common
import image_dataset as ds

print("Just test")

import numpy
import tensorflow as tf
from random import randint

if __name__ == "__main__":

    dims = 8
    pos = randint(0, dims - 1)

    logits = tf.random_uniform([dims], maxval=3, dtype=tf.float32)
    labels = tf.one_hot(pos, dims)

    res1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    res2 = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=tf.constant(pos))

    with tf.Session() as sess:
        a, b = sess.run([res1, res2])
        print(a, b)
        print(a == b)
