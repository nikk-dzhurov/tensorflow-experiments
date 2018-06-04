import os
import sys
import json
import pickle
import urllib
import tarfile
import tensorflow as tf


def clean_dir(dir_name):
    """Clean directory's content"""

    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)


def save_json(data, filename):
    """Save data in JSON file"""

    with open(filename, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
        print("%s saved" % filename)


def load_json(filename):
    """Load data from JSON file"""

    with open(filename, 'r') as fp:
        return json.load(fp)


def save_pickle(data, filename):
    """Save data in pickle file"""

    with open(filename, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)
        print("%s saved" % filename)


def load_pickle(filename):
    """Load data from pickle file"""

    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def _track_progress(count, block_size, total_size):
    """Progress report function"""

    progress = float(count * block_size) / float(total_size) * 100.0
    sys.stdout.write('\r\t>> Progress %.2f%%' % progress)
    sys.stdout.flush()


def maybe_download_and_extract(dest_dir, data_url, nested_dir):
    """
    Download and extract dataset if directory does not exist
    Example usage for cifar10 dataset:
        maybe_download_and_extract(
            dest_dir="./test-data",
            data_url='https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
            nested_dir=os.path.join(os.getcwd(), "test-data/cifar-10-batches-bin")
        )
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)

    if not os.path.exists(filepath):
        print("\nDownloading %s" % filename)
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _track_progress)
        statinfo = os.stat(filepath)
        print('\nSuccessfully downloaded %s : %d bytes.' % (filename, statinfo.st_size))
    else:
        statinfo = os.stat(filepath)
        print('\nFile is already downloaded %s : %d bytes.' % (filename, statinfo.st_size))

    extracted_dir = os.path.join(dest_dir, nested_dir)
    if not os.path.exists(extracted_dir):
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)
        print('\File %s is extracted successfully at %s' % (filename, extracted_dir))
    else:
        print('File %s is already extracted successfully at %s' % (filename, extracted_dir))
