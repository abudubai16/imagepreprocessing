import cv2
import numpy as np

# have to run the run functions using multiprocessing for faster processing


class Resize:
    fx: float
    fy: float
    dsize: tuple[int]
    interpolation: int

    def __init__(self, fx: float, fy: float, dsize: tuple[int] = (256, 256), interpolation: int = cv2.INTER_AREA):
        self.dsize = dsize
        self.fx = fx
        self.fy = fy
        self.interpolation = interpolation

    def run(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        pass


class Color:
    code: int

    def __init__(self, code: int, dst: int, dstCn: int):
        pass

    def run(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.cvtColor
        return images


class Normalize:
    def __init__(self):
        pass

    def run(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.normalize
        return images


class Blur:
    type: int

    def __init__(self, type: int):
        self.type = type
        blur = [self.box_filter, self.simple_blur]

    def run(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.GaussianBlur
        return images

    def box_filter(self):
        pass

    def simple_blur(self):
        pass

class Filter2d:
    def __init__(self):
        pass

    def run(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.filter2D
        return images
