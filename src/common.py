import os
import sys
import urllib
import tarfile
import tensorflow as tf


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var.name.replace(":", "_")):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def build_layer_summaries(layer_name):
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=layer_name):
        variable_summaries(var)


def clean_dir(dir_name):
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)


def _track_progress(count, block_size, total_size):
    progress = float(count * block_size) / float(total_size) * 100.0
    sys.stdout.write('\r\t>> Progress %.2f%%' % progress)
    sys.stdout.flush()


def maybe_download_and_extract(dest_dir, data_url, nested_dir=""):
    """Download and extract data from tarball"""
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