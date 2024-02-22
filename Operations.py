import cv2


def resize(func=cv2.resize, *kwargs):
    return func, kwargs


def grayscale(func=cv2.cvtColor, *kwargs):
    return func, kwargs


def normalize(func=cv2.normalize, *kwargs):
    return func, kwargs


def blur(blur_type=cv2.GaussianBlur, *kwargs):
    return blur_type, kwargs


def filter_kernal(func=cv2.filter2D, *kwargs):
    return func, kwargs


def custom_function(func, *kwargs):
    return func, kwargs
