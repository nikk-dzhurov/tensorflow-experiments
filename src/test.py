from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import clean_dir
from common import save_pickle
from common import load_pickle
from common import maybe_download_and_extract

import os
from PIL import Image
import numpy as np
import tensorflow as tf

# maybe_download_and_extract(
#     dest_dir=os.path.join(os.getcwd(), "test-data"),
#     data_url='https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
#     nested_dir='cifar-10-batches-bin'
# )

RANDOM_SEED = 55355


def randomly_distort_image(img, crop_shape=(26, 26, 3), target_width=32, seed=None, target_height=32):
    dist = tf.random_crop(img, crop_shape, seed=seed)
    dist = tf.image.random_contrast(dist, lower=0.3, upper=1.7, seed=seed)
    dist = tf.image.random_hue(dist, max_delta=0.2, seed=seed)
    dist = tf.image.random_flip_left_right(dist, seed=seed)

    return tf.image.resize_image_with_crop_or_pad(dist, target_width, target_height)


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

    def save_image(self, location=None, name=None):
        if self.image is None:
            raise RuntimeError("Image data is missing")

        if name is None:
            if self.name is None:
                raise RuntimeError("Image's name is missing")
            name = "%s.jpg" % self.name

        if location is None:
            location = os.getcwd()

        img = Image.fromarray(self.image, "RGB")
        img.save(os.path.join(location, name))


class Dataset(object):
    def __init__(self, data, labels):
        self.x = data
        self.y = labels

    def save_to_pickle(self, name="dataset.pkl"):
        save_pickle(self, name)

    def append(self, data, labels):
        if len(data) != len(labels):
            raise RuntimeError("Length of data and labels should be equal")

        self.x = np.append(self.x, data)
        self.y = np.append(self.y, labels)

    def mirror_images(self):
        with tf.Session() as sess:
            mirror_x = sess.run(tf.map_fn(
                fn=lambda img: tf.image.flip_left_right(img),
                elems=self.x,
                parallel_iterations=100
            ))

            return Dataset(mirror_x, self.y.copy())

    def randomly_distort_images(self, seed=None):
        with tf.Session() as sess:
            random_dist_data = sess.run(tf.map_fn(
                fn=lambda img: randomly_distort_image(img, seed=seed),
                elems=self.x,
                parallel_iterations=100
            ))

            return Dataset(random_dist_data, self.y.copy())


def improve_dataset(train, test, dataset_name="dataset_name", seed=None, save_location=None):
    # EXAMPLE USAGE FOR CIFAR10
    # train, test = tf.keras.datasets.cifar10.load_data()
    # improve_dataset(train, test, "cifar10", seed=RANDOM_SEED, save_location="../datasets")

    train_x, train_y = train
    test_x, test_y = test

    if save_location is None:
        save_location = os.getcwd()

    save_location = os.path.join(save_location, dataset_name)

    # clean old data
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    clean_dir(save_location)

    # original data
    test_ds = Dataset(test_x, test_y)
    train_ds = Dataset(train_x, train_y)

    test_ds.save_to_pickle(
        os.path.join(save_location, "original_test.pkl"))
    train_ds.save_to_pickle(
        os.path.join(save_location, "original_train.pkl"))

    # mirror images in the dataset
    mirror_ds = train_ds.mirror_images()
    mirror_ds.save_to_pickle(
        os.path.join(save_location, "mirror_train.pkl"))

    # randomly distort images in the dataset

    random_dist_ds = train_ds.randomly_distort_images(seed=seed)
    random_dist_ds.save_to_pickle(
        os.path.join(save_location,"rand_distorted_train.pkl"))


def load_dataset_from_pickles(pickle_locations):
    # EXAMPLE USAGE FOR CIFAR10
    # pickles = [
    #     "/datasets/cifar10/original_train.pkl",
    #     "/datasets/cifar10/mirror_train.pkl",
    #     "/datasets/cifar10/rand_distorted_train.pkl",
    # ]
    #
    # ds = load_dataset_from_pickles(pickles)
    #
    # # # load and save image
    # offset = 555
    # img0 = LabeledImage().load_from_dataset_tuple((ds.x, ds.y), 0 + offset)
    # img1 = LabeledImage().load_from_dataset_tuple((ds.x, ds.y), 50000 + offset)
    # img2 = LabeledImage().load_from_dataset_tuple((ds.x, ds.y), 100000 + offset)
    #
    # mixed_img = np.concatenate([img0.image, img1.image, img2.image], axis=1)
    # LabeledImage(mixed_img, "mixed_" + img0.name) \
    #     .save_image()

    result = None
    length = len(pickle_locations)
    for i in range(length):
        ds = load_pickle(pickle_locations[i])
        if result is None:
            result = ds
        else:
            result.x = np.concatenate([result.x, ds.x], axis=0)
            result.y = np.concatenate([result.y, ds.y], axis=0)

    return result

