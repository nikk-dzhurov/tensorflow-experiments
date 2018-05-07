from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
import tensorflow as tf

import common
import image_dataset as ds

RANDOM_SEED = 55355

train, test = common.load_original_stl10()


# ds.improve_dataset(train, test, "stl10", crop_shape=(72, 72, 3), target_size=96, rand_dist_sets=3, seed=RANDOM_SEED, save_location="../datasets")

