import cv2
import numpy as np


class Resize:

    def __init__(self, fx: float, fy: float, dsize: tuple[int] = (256, 256), interpolation: int = cv2.INTER_AREA):
        self.dsize = dsize
        self.fx = fx
        self.fy = fy
        self.interpolation = interpolation

    def run(self, image: np.ndarray, bounding_boxes: np.ndarray) -> np.ndarray:
        img_shape = np.shape(image)[:-1]
        return image, bounding_boxes


class Color:

    def __init__(self, code: int):
        self.code = code

    def run(self, image: np.ndarray) -> np.ndarray:
        print(3)
        image = cv2.cvtColor(image, code=self.code)
        print(4)
        return image


class Normalize:
    def __init__(self, alpha: float, beta: float, norm_type: int = cv2.NORM_MINMAX):
        self.norm_type = norm_type
        self.alpha = alpha
        self.beta = beta

    def run(self, image: np.ndarray) -> np.ndarray:
        image = cv2.normalize(image,
                              alpha=self.alpha,
                              beta=self.beta,
                              norm_type=self.norm_type
                              )
        return image


class Blur:

    def __init__(self,
                 blur_type: int,
                 ksize: np.ndarray | None,
                 ddepth: int = 3,
                 sigmaY: float = 0,
                 sigmaX: float = 1,
                 borderType: int = cv2.BORDER_DEFAULT,
                 normalize: bool = False,
                 sigmaColor: float = 75,
                 sigmaSpace: float = 75,
                 d: int = 15
                 ):

        self.blurType = blur_type
        self.ksize = ksize
        self.ddepth = ddepth
        self.sigmaX = sigmaX
        self.sigmaY = sigmaX if sigmaY == 0 else sigmaY
        self.borderType = borderType
        self.normalize = normalize
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
        self.d = d

    def run(self, image: np.ndarray) -> np.ndarray:
        if self.blurType == 0:
            images = self.box_filter(image=image)
        elif self.blurType == 1:
            images = self.gaussian_blur(image=image)
        elif self.blurType == 2:
            images = self.simple_blur(image=image)
        elif self.blurType == 3:
            images = self.median_blur(image=image)
        elif self.blurType == 4:
            images = self.bilateral_blur(image=image)
        elif self.blurType == 5:
            images = self.SobelFilter(image=image)
        else:
            raise ValueError
        return images

    def gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        image = cv2.GaussianBlur(image,
                                 ksize=self.ksize,
                                 sigmaX=self.sigmaX,
                                 sigmaY=self.sigmaY,
                                 borderType=self.borderType
                                 )
        return image

    def box_filter(self, image: np.ndarray) -> np.ndarray:
        image = cv2.boxFilter(image,
                              ddepth=self.ddepth,
                              ksize=self.ksize,
                              normalize=self.normalize,
                              borderType=self.borderType
                              )
        return image

    def simple_blur(self, image: np.ndarray) -> np.ndarray:
        image = cv2.blur(image,
                         ksize=self.ksize,
                         borderType=self.borderType
                         )
        return image

    def median_blur(self, image: np.ndarray) -> np.ndarray:
        image = cv2.medianBlur(image,
                               ksize=self.ksize
                               )
        return image

    def bilateral_blur(self, image: np.ndarray) -> np.ndarray:
        image = cv2.bilateralFilter(image,
                                    d=self.d,
                                    sigmaColor=self.sigmaColor,
                                    sigmaSpace=self.sigmaSpace
                                    )
        return image

    def SobelFilter(self, image: np.ndarray) -> np.ndarray:
        SOBELKERNAL = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        img_x = cv2.filter2D(image, cv2.CV_64F, kernel=SOBELKERNAL)
        img_y = cv2.filter2D(image, cv2.CV_64F, kernel=SOBELKERNAL.transpose)
        img = cv2.normalize(np.sqrt(img_x**2+img_y**2), None, 0, 255, cv2.NORM_MINMAX)
        return img


class Filter2d:

    def __init__(self, ddpth: int, kernal: np.ndarray, delta: float = 0, borderType: int = cv2.BORDER_DEFAULT):
        self.ddpth = ddpth
        self.kernal = kernal
        self.delta = delta
        self.borderType = borderType

    def run(self, image: np.ndarray) -> np.ndarray:
        image = cv2.filter2D(image,
                             ddepth=self.ddpth,
                             delta=self.delta,
                             borderType=self.borderType,
                             kernel=self.kernal
                             )
        return image


class CustomFunc:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self, image: np.ndarray) -> np.ndarray:
        image = self.func(image, *self.args, **self.kwargs)
        return image

