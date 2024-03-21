import unittest
import os

import cv2
import numpy as np

from .. import Operations
from .. import ImgData


class TestModules(unittest.TestCase):

    def test_basic_functions(self):
        dirname = os.path.dirname(__file__)
        imgs_dir = f"{dirname}\images"

        img_name = ["bunny.jpg"]
        imgs_dataset = ImgData.Image(img_name, directory=imgs_dir)

        model = ImgData.Sequential([
            Operations.Color(cv2.COLOR_BGR2GRAY),
            Operations.HistEqualize()
        ])
        model.process(imgs_dataset)

        test_prediction = cv2.imread(f"{dirname}\images\processed_images\{img_name[0]}")
        test_result = cv2.imread(f"{dirname}\img_results\{img_name[0]}")

        self.assertEqual(np.array_equal(test_prediction, test_result), True)


if __name__ == "__main__":
    unittest.main()
