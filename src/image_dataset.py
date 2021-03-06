import os
import random
import numpy as np
import tensorflow as tf

import file

RANDOM_SEED = 55355


class ImageDataset(object):
    """
    ImageDataset class for loading/exporting image datasets
    Also could be used for distorting the whole dataset at once

    It supports the following distortions:
        * mirror images
        * rotate images by 90 degrees
        * distort images by modifying contrast
    """

    def __init__(self, data, labels):
        """Initialize/Construct ImageDataset object"""

        self.x = data
        self.y = labels

        self.DIST_BATCH_SIZE = 2000
        self.PARALLEL_ITERATIONS = 150

    @staticmethod
    def load_from_pickles(pickle_locations):
        """Load multiple datasets from pickle_locations"""

        result = None
        length = len(pickle_locations)
        for i in range(length):
            ds = file.load_pickle(pickle_locations[i])
            if result is None:
                result = ds
            else:
                result.x = np.concatenate([result.x, ds.x], axis=0)
                result.y = np.concatenate([result.y, ds.y], axis=0)

        return result

    def save_to_pickle(self, name="dataset.pkl"):
        """Export dataset to pickle file"""

        file.save_pickle(self, name)

    def append(self, data, labels):
        """Append another dataset to current object"""

        if len(data) != len(labels):
            raise RuntimeError("Length of data and labels should be equal")

        self.x = np.append(self.x, data)
        self.y = np.append(self.y, labels)

    def mirror_images(self):
        """Build another ImageDataset by mirroring the images"""

        return self._distort_on_batches(lambda batch, parallel_iter: tf.compat.v1.Session().run(
            tf.map_fn(
                fn=lambda img: tf.image.flip_left_right(img),
                elems=batch,
                parallel_iterations=parallel_iter
            )
        ))

    def rot90_images(self, n_times=1):
        """Build another ImageDataset by rotating the images by 90 degrees, N times"""

        return self._distort_on_batches(lambda batch, _: tf.compat.v1.Session().run(
            tf.image.rot90(batch, k=n_times)
        ))

    def randomly_distort_images(self, crop_shape, target_size, seed=None):
        """Build another ImageDataset by distorting the images"""

        return self._distort_on_batches(lambda batch, parallel_iter: tf.compat.v1.Session.run(
            tf.map_fn(
                fn=lambda img: self._randomly_distort_image(
                    image=img,
                    crop_shape=crop_shape,
                    target_size=target_size,
                    seed=seed
                ),
                elems=batch,
                parallel_iterations=parallel_iter
            )
        ))

    def _distort_on_batches(self, dist_fn):
        """Base function for creating new ImageDataset by distorting current ImageDataset on batches"""

        random_dist_data = None
        data_len = len(self.x)
        steps = data_len // self.DIST_BATCH_SIZE
        if data_len % self.DIST_BATCH_SIZE:
            steps += 1

        for step in range(steps):
            print("Step {} of {}".format(step, steps))

            batch = self._get_current_batch(step)

            partial_data = dist_fn(batch, self.PARALLEL_ITERATIONS)

            if random_dist_data is None:
                random_dist_data = partial_data
            else:
                random_dist_data = np.concatenate([random_dist_data, partial_data], axis=0)

        print(random_dist_data.shape)

        return ImageDataset(random_dist_data, self.y.copy())

    def _get_current_batch(self, curr_step):
        """Get current batch, depending on current step and default BATCH_SIZE"""

        start_idx = curr_step * self.DIST_BATCH_SIZE
        end_idx = start_idx + self.DIST_BATCH_SIZE

        return self.x[start_idx:end_idx]

    @staticmethod
    def _randomly_distort_image(image, crop_shape, target_size, seed=None):
        """
        Distort image by random factor
        This function should be used inside tf.compat.v1.Session
        """

        dist = tf.random_crop(image, crop_shape, seed=seed)
        dist = tf.image.random_contrast(dist, lower=0.7, upper=1.3, seed=seed)
        dist = tf.image.random_brightness(dist, max_delta=0.3, seed=seed)
        dist = tf.image.random_flip_left_right(dist, seed=seed)

        return tf.image.resize_image_with_crop_or_pad(dist, target_size, target_size)


def split_dataset(images, labels, classes_count,
                  test_items_per_class=None, test_items_fraction=None):
    """Split dataset to test and train by given fraction or given number of items per class for test dataset"""

    if test_items_per_class is None and test_items_fraction is None:
        raise ValueError("Please specify test_items_per_class or test_items_fraction")
    if test_items_per_class is not None and test_items_fraction is not None:
        raise ValueError("Please specify either test_items_per_class or test_items_fraction")

    test_ds = []
    indices = []
    count_per_class = np.zeros(classes_count, int)

    if test_items_per_class is None:
        test_items_per_class = int(labels.shape[0] * test_items_fraction / 10)

    zipped_ds = _zip_ds_pairs(images, labels)

    random.shuffle(zipped_ds)

    for idx, pair in enumerate(zipped_ds):
        label = pair[0]
        if count_per_class[label] < test_items_per_class:
            test_ds.append(pair)
            count_per_class[label] += 1
            indices.append(idx)

    for i in sorted(indices, reverse=True):
        del zipped_ds[i]

    train_ds = _unzip_ds_pairs(zipped_ds)
    test_ds = _unzip_ds_pairs(test_ds)

    return train_ds, test_ds


def improve_dataset(train, test, dataset_name, crop_shape,
                    target_size, add_rot90_dist=False, rand_dist_sets=1,
                    seed=RANDOM_SEED, save_location=None):
    """Function that applies multiple distortions over original dataset"""

    test_x, test_y = test
    train_x, train_y = train

    if save_location is None:
        save_location = os.getcwd()

    save_location = os.path.join(save_location, dataset_name)

    # clean old data
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    file.clean_dir(save_location)

    # original data
    test_ds = ImageDataset(test_x, test_y)
    train_ds = ImageDataset(train_x, train_y)

    # save original dataset to pickle file
    test_ds.save_to_pickle(
        os.path.join(save_location, "original_test.pkl"))
    del test_ds

    train_ds.save_to_pickle(
        os.path.join(save_location, "original_train.pkl"))

    # mirror images in the dataset and save them to pickle file
    train_ds.mirror_images().save_to_pickle(
        os.path.join(save_location, "mirror_train.pkl"))

    if add_rot90_dist:
        # rotate images in the dataset by 90 degrees 1 time and save them to pickle file
        train_ds.rot90_images(1).save_to_pickle(
            os.path.join(save_location, "rot_90_1_train.pkl"))

        # rotate images in the dataset by 90 degrees 3 times and save them to pickle file
        train_ds.rot90_images(3).save_to_pickle(
            os.path.join(save_location, "rot_90_3_train.pkl"))

    # randomly distort images in the dataset and save them to pickle file
    if rand_dist_sets >= 1:
        for i in range(rand_dist_sets):
            train_ds.randomly_distort_images(
                seed=seed + i, crop_shape=crop_shape, target_size=target_size) \
                .save_to_pickle(
                os.path.join(save_location, "rand_distorted_train_%d.pkl" % i))

    print("Improving dataset is completed")


def prepare_images(images, dtype=np.float32):
    """
    Convert type of images' data(int8) to np.float32(by default)
    Convert data values from range [0,255] to [0, 1] (preparation for model train/eval/predict)
    """
    images = np.asarray(images, dtype=dtype)

    return np.multiply(images, 1.0 / 255.0)


def _zip_ds_pairs(images, labels):
    """Zip images and labels in single list"""
    zipped_ds = []

    for idx, label in enumerate(labels):
        zipped_ds.append([label, images[idx]])

    return zipped_ds


def _unzip_ds_pairs(ds):
    """Unzip images and labels from single list"""
    ds_x = []
    ds_y = []

    for pair in ds:
        ds_y.append(pair[0])
        ds_x.append(pair[1])

    return np.asarray(ds_x), np.asarray(ds_y)
