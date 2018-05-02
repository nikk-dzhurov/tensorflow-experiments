import os
import numpy as np
import tensorflow as tf

import common
import image


class ImageDataset(object):
    def __init__(self, data, labels):
        self.x = data
        self.y = labels

    def save_to_pickle(self, name="dataset.pkl"):
        common.save_pickle(self, name)

    def append(self, data, labels):
        if len(data) != len(labels):
            raise RuntimeError("Length of data and labels should be equal")

        self.x = np.append(self.x, data)
        self.y = np.append(self.y, labels)

    def mirror_images(self):
        with tf.Session() as sess:
            batch_size = 5000
            mirror_x = None
            data_len = len(self.x)
            steps = data_len // batch_size
            if data_len % batch_size:
                steps += 1

            for i in range(steps):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                if data_len >= end_idx:
                    batch = self.x[start_idx:end_idx]
                else:
                    batch = self.x[start_idx:-1]

                partial_data = sess.run(tf.map_fn(
                    fn=lambda img: tf.image.flip_left_right(img),
                    elems=batch,
                    parallel_iterations=1000
                ))

                if mirror_x is None:
                    mirror_x = partial_data
                else:
                    mirror_x = np.concatenate([mirror_x, partial_data], axis=0)

            return ImageDataset(mirror_x, self.y.copy())

    def randomly_distort_images(self, seed=None):
        with tf.Session() as sess:
            batch_size = 5000
            random_dist_data = None
            data_len = len(self.x)
            steps = data_len // batch_size
            if data_len % batch_size:
                steps += 1

            for i in range(steps):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                if data_len >= end_idx:
                    batch = self.x[start_idx:end_idx]
                else:
                    batch = self.x[start_idx:-1]

                partial_data = sess.run(tf.map_fn(
                    fn=lambda img: image.randomly_distort_image(img, seed=seed),
                    elems=batch,
                    parallel_iterations=1000
                ))

                if random_dist_data is None:
                    random_dist_data = partial_data
                else:
                    random_dist_data = np.concatenate([random_dist_data, partial_data], axis=0)

            return ImageDataset(random_dist_data, self.y.copy())


def improve_dataset(train, test, dataset_name="dataset_name", seed=None, save_location=None):
    # EXAMPLE USAGE FOR CIFAR10
    # train, test = tf.keras.datasets.cifar10.load_data()
    # improve_dataset(train, test, "cifar10", seed=RANDOM_SEED, save_location="../datasets")

    test_x, test_y = test
    train_x, train_y = train

    if save_location is None:
        save_location = os.getcwd()

    save_location = os.path.join(save_location, dataset_name)

    # clean old data
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    common.clean_dir(save_location)

    # original data
    test_ds = ImageDataset(test_x, test_y)
    train_ds = ImageDataset(train_x, train_y)

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
        os.path.join(save_location, "rand_distorted_train.pkl"))

    print("Improving dataset is completed")


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
        ds = common.load_pickle(pickle_locations[i])
        if result is None:
            result = ds
        else:
            result.x = np.concatenate([result.x, ds.x], axis=0)
            result.y = np.concatenate([result.y, ds.y], axis=0)

    return result
