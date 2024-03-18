import os
import numpy as np

from ..errors import image_errors


class Image:
    def __init__(self, img_id: np.ndarray | str, directory: str, **kwargs):
        self.img_id = img_id
        self.directory = directory
        self.b_boxes = None
        if "bb" in kwargs:
            self.b_boxes = kwargs.get("bb")

        if not os.path.exists(self.directory):
            raise NotADirectoryError

    def __add__(self, other):
        self.img_id = np.concatenate(self.img_id, other.img_id, axis=1)
        self.b_boxes = np.concatenate(self.b_boxes, other.b_boxes, axis=1)

        if self.directory != other.directory:
            raise image_errors.DirectoryNotSame

        return self

    def __repr__(self):
        if int(np.shape(self.img_id)[0]) >= 5:
            message = f"Number of images:{int(np.shape(self.img_id)[0])}\nThe first 5 paths are:"
            for i in range(5):
                message = message + f"\n{self.img_id[i]}"
        else:
            message = f"Number of images:{int(np.shape(self.img_id)[0])}\nThe first few paths are:"
            for path in self.img_id:
                message = message + f"\n{path}"
        return message
