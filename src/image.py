import os
import numpy as np
from PIL import Image
import tensorflow as tf


class LabeledImage(object):
    def __init__(self, image, name="image", max_value=1):
        self.image = image
        self.max_value = max_value
        self.name = str(name) + ".jpg"

    @staticmethod
    def load_from_dataset_tuple(dataset, index=0, max_value=1):
        if dataset is None:
            raise ValueError("Invalid initialization parameters provided")

        return LabeledImage(
            image=dataset[0][index],
            name=str(dataset[1][index]) + ".jpg",
            max_value=max_value,
        )

    def save(self, location=None, name=None, ):
        if self.image is None:
            raise ValueError("Image data is missing")

        if name is None:
            name = self.name

        if location is None:
            location = os.getcwd()

        self.normalize()

        # save image data
        img = Image.fromarray(self.image, "RGB")
        img.save(os.path.join(location, name))

    def normalize(self):
        if self.max_value != 255:
            self.image = np.multiply(self.image, 255.0 / self.max_value)
            self.image = np.asarray(self.image, dtype=np.int8)
            self.max_value = 255


def randomly_distort_image(image, crop_shape=(26, 26, 3), target_size=32, seed=None):
    dist = tf.random_crop(image, crop_shape, seed=seed)
    dist = tf.image.random_contrast(dist, lower=0.7, upper=1.3, seed=seed)
    dist = tf.image.random_hue(dist, max_delta=0.1, seed=seed)
    dist = tf.image.random_flip_left_right(dist, seed=seed)

    return tf.image.resize_image_with_crop_or_pad(dist, target_size, target_size)
