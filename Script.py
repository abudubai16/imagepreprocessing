import Image
import cv2


def script():
    image = cv2.imread("Buddha.jpg")
    print(type(image))
