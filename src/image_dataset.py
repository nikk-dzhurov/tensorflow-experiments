import os
import numpy as np
import tensorflow as tf

import common
import image

RANDOM_SEED = 55355

class ImageDataset(object):
    def __init__(self, data, labels):
        self.x = data
        self.y = labels

        self.dist_batch_size = 500
        self.parallel_iterations = 100

    @staticmethod
    def load_from_pickles(pickle_locations):
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

    def save_to_pickle(self, name="dataset.pkl"):
        common.save_pickle(self, name)

    def append(self, data, labels):
        if len(data) != len(labels):
            raise RuntimeError("Length of data and labels should be equal")

        self.x = np.append(self.x, data)
        self.y = np.append(self.y, labels)

    def mirror_images(self):
        with tf.Session() as sess:
            mirror_x = None
            data_len = len(self.x)
            steps = data_len // self.dist_batch_size
            if data_len % self.dist_batch_size:
                steps += 1

            for step in range(steps):
                batch = self._get_current_batch(step)

                partial_data = sess.run(tf.map_fn(
                    fn=lambda img: tf.image.flip_left_right(img),
                    elems=batch,
                    parallel_iterations=self.parallel_iterations
                ))

                if mirror_x is None:
                    mirror_x = partial_data
                else:
                    mirror_x = np.concatenate([mirror_x, partial_data], axis=0)

            print(mirror_x.shape)

            return ImageDataset(mirror_x, self.y.copy())

    def randomly_distort_images(self, crop_shape, target_size, seed=None):
        with tf.Session() as sess:
            random_dist_data = None
            data_len = len(self.x)
            steps = data_len // self.dist_batch_size
            if data_len % self.dist_batch_size:
                steps += 1

            for step in range(steps):
                batch = self._get_current_batch(step)

                partial_data = sess.run(tf.map_fn(
                    fn=lambda img: image.randomly_distort_image(
                        img, crop_shape=crop_shape, target_size=target_size, seed=seed),
                    elems=batch,
                    parallel_iterations=self.parallel_iterations
                ))

                if random_dist_data is None:
                    random_dist_data = partial_data
                else:
                    random_dist_data = np.concatenate([random_dist_data, partial_data], axis=0)

            print(random_dist_data.shape)

            return ImageDataset(random_dist_data, self.y.copy())

    def _get_current_batch(self, curr_step):
        start_idx = curr_step * self.dist_batch_size
        end_idx = start_idx + self.dist_batch_size

        return self.x[start_idx:end_idx]


def improve_dataset(train, test, dataset_name="dataset_name", crop_shape=(26, 26, 3), target_size=32, rand_dist_sets=1, seed=RANDOM_SEED, save_location=None):
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
    train_ds.mirror_images().save_to_pickle(
        os.path.join(save_location, "mirror_train.pkl"))

    # randomly distort images in the dataset
    if rand_dist_sets > 1:
        for i in range(rand_dist_sets):
            train_ds.randomly_distort_images(
                seed=seed+i, crop_shape=crop_shape, target_size=target_size)\
                .save_to_pickle(
                os.path.join(save_location, "rand_distorted_train_%d.pkl" % i))
    else:
        train_ds.randomly_distort_images(
            seed=seed, crop_shape=crop_shape, target_size=target_size)\
            .save_to_pickle(
            os.path.join(save_location, "rand_distorted_train.pkl"))

    print("Improving dataset is completed")


