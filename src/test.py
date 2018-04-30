from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import clean_dir
from common import build_layer_summaries
from common import maybe_download_and_extract

import os


maybe_download_and_extract(
    dest_dir=os.path.join(os.getcwd(), "test-data"),
    data_url='https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
    nested_dir='cifar-10-batches-bin'
)
