import os
import numpy as np
from PIL import Image
import tensorflow as tf


class LabeledImage(object):
    def __init__(self, image=None, name=None):
        self.image = image
        if name is not None:
            self.name = str(name)
        else:
            self.name = None

    def load_from_dataset_tuple(self, dataset, index=0):
        if dataset is not None:
            self.image = dataset[0][index]
            self.name = str(dataset[1][index])
        else:
            raise RuntimeError("Invalid initialization parameters provided")

        return self

    def save_image(self, location=None, name=None, ):
        if self.image is None:
            raise RuntimeError("Image data is missing")

        if name is None:
            if self.name is None:
                raise RuntimeError("Image's name is missing")
            name = "%s.jpg" % self.name

        if location is None:
            location = os.getcwd()

        # normalize image data
        img = np.multiply(self.image, 255.0)
        img = np.asarray(img, dtype=np.int8)

        # save image data
        img = Image.fromarray(img, "RGB")
        img.save(os.path.join(location, name))


def randomly_distort_image(img, crop_shape=(26, 26, 3), target_width=32, seed=None, target_height=32):
    dist = tf.random_crop(img, crop_shape, seed=seed)
    dist = tf.image.random_contrast(dist, lower=0.7, upper=1.3, seed=seed)
    dist = tf.image.random_hue(dist, max_delta=0.1, seed=seed)
    dist = tf.image.random_flip_left_right(dist, seed=seed)

    return tf.image.resize_image_with_crop_or_pad(dist, target_width, target_height)
