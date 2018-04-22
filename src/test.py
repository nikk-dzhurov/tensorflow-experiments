import os
import common


common.maybe_download_and_extract(
    dest_dir=os.path.join(os.getcwd(), "test-data"),
    data_url='https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
    nested_dir='cifar-10-batches-bin'
)

# common.maybe_download_and_extract()