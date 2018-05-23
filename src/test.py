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



if __name__ == "__main__":
    arr = [5., 4., 6., 7., 5.5, 6.5, 4.5, 4.],


    # tf.reshape(arr, (1))

    res = tf.Session().run(tf.nn.softmax(arr, axis=-1))
    print(res)