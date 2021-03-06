import os
import numpy as np
from PIL import Image

from image_dataset import ImageDataset


class LabeledImage(object):
    """LabeledImage class for exporting images"""

    def __init__(self, image, name="image", max_value=1):
        """Initialize/Construct LabeledImage object from image data"""

        self.image = image
        self.max_value = max_value
        if type(name) is str and name.endswith(".jpg"):
            self.name = name
        else:
            self.name = str(name) + ".jpg"

    @staticmethod
    def load_from_dataset(dataset, index=0, max_value=1):
        """Construct LabeledImage object from dataset and image's index in dataset"""

        if dataset is None:
            raise ValueError("Invalid initialization parameters provided")

        if isinstance(dataset, ImageDataset):
            dataset = (dataset.x, dataset.y)

        return LabeledImage(
            image=dataset[0][index],
            name=str(dataset[1][index]) + ".jpg",
            max_value=max_value,
        )

    def save(self, location=None, name=None):
        """Save image in location"""

        if self.image is None:
            raise ValueError("Image data is missing")

        if name is None:
            name = self.name

        if location is None:
            location = os.getcwd()

        self.normalize()

        # save image data(JPEG format)
        img = Image.fromarray(self.image, "RGB")
        img.save(os.path.join(location, name))

    def normalize(self):
        """Normalize image values from range [0, 1] to [0, 255] if necessary"""

        if self.max_value != 255:
            self.image = np.multiply(self.image, 255.0 / self.max_value)
            self.image = np.asarray(self.image, dtype=np.int8)
            self.max_value = 255
