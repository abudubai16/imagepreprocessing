import cv2
import numpy as np
import operationerrors
# have to run the run functions using multiprocessing for faster processing


class Resize:

    def __init__(self, fx: float, fy: float, dsize: tuple[int] = (256, 256), interpolation: int = cv2.INTER_AREA):
        self.dsize = dsize
        self.fx = fx
        self.fy = fy
        self.interpolation = interpolation

    def run(self, images: np.ndarray, bounding_boxes: np.ndarray) -> np.ndarray[np.ndarray]:

        return images, bounding_boxes


class Color:

    def __init__(self, code: int, dst: int, dstCn: int):
        self.code = code
        self.dst = dst
        self.dstCn = dstCn

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

    def __init__(self, blur_type: int, borderType: int = cv2.BORDER_DEFAULT):
        self.blurType = blur_type
        self.blur_functions = [
                self.box_filter,
                self.gaussian_blur,
                self.simple_blur,
                self.median_blur,
                self.bilateral_blur
            ]

    def run(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        if self.blurType == 0:
            images = self.box_filter(images=images)
        elif self.blurType == 1:
            images = self.gaussian_blur(images=images)
        elif self.blurType == 2:
            images = self.simple_blur(images=images)
        elif self.blurType == 3:
            images = self.median_blur(images=images)
        elif self.blurType == 4:
            images = self.bilateral_blur(images=images)
        else:
            raise ValueError
        return images

    def gaussian_blur(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.GaussianBlur
        return images

    def box_filter(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.boxFilter
        return images

    def simple_blur(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.blur
        return images

    def median_blur(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.medianBlur
        return images

    def bilateral_blur(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.bilateralFilter
        return images


class Filter2d:

    def __init__(self, ddpth: int, kernal: np.ndarray, delta: float = 0, borderType: int = cv2.BORDER_DEFAULT):
        self.ddpth = ddpth
        self.kernal = kernal
        self.delta = delta
        self.borderType = borderType

    def run(self, images: np.ndarray) -> np.ndarray[np.ndarray]:
        func = cv2.filter2D
        return images
