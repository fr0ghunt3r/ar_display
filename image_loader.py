import os
import cv2
import time
from abc import abstractmethod


class ImageLoader:
    extensions: tuple = \
        (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".tfrecords")

    def __init__(self, path: str, mode: str = "BGR"):
        self.path = path
        self.mode = mode
        self.dataset = self.parse_input(self.path)
        self.sample_idx = 0

    def parse_input(self, path):

        # single image or tfrecords file
        if os.path.isfile(path):
            assert path.lower().endswith(self.extensions,), f"Unsupportable extension, please, use one of {self.extensions}"
            return [path]

        if os.path.isdir(path):
            # lmdb environment
            if any([file.endswith(".mdb") for file in os.listdir(path)]):
                return path
            else:
                # folder with images
                paths = \
                    [os.path.join(path, image) for image in os.listdir(path) if self.extensions[1] == image[-4:]]
                paths.sort()
                return paths

    def __iter__(self):
        self.sample_idx = 0
        return self

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def __next__(self):
        pass

class CV2Loader(ImageLoader):
    def __next__(self):
        start = time.time()
        # get image path by index from the dataset
        path = self.dataset[self.sample_idx]
        # read the image
        image = cv2.imread(path)
        full_time = time.time() - start

        if self.mode == "RGB":
            start = time.time()
            # change color mode
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            full_time += time.time() - start

        self.sample_idx += 1
        return image, full_time
