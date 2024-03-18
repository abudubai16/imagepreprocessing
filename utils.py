import cv2
import numpy as np

# TODO finish stuff here


def get_chunksize(num_img: int, num_cores: int) -> int:
    return num_img//num_cores


def resize_bb(bb: np.ndarray, new_size: tuple, curr_size: tuple) -> np.ndarray:
    return bb


def create_points(bounding_boxes: np.ndarray, bb_format: str) -> list:
    points = []
    num_points = 4
    if bb_format == "xyxy":
        pass
    elif bb_format == "xyxyxyxy":
        num_points = 6
    elif bb_format == "xywh":
        t = np.zeros(np.shape(bounding_boxes))
        t[:, 0] = bounding_boxes[:, 0] + bounding_boxes[:, 2]/2
        t[:, 2] = bounding_boxes[:, 2] - bounding_boxes[:, 2] / 2
        t[:, 1] = bounding_boxes[:, 1] + bounding_boxes[:, 1] / 2
        t[:, 3] = bounding_boxes[:, 3] - bounding_boxes[:, 1] / 2
        bounding_boxes = t
    else:
        pass

    for i in range(num_points):
        points.append([[(bounding_boxes[i % num_points],
                         bounding_boxes[(i + 1) % num_points])] for i in range(num_points)])

    points = np.concatenate(tuple(points), axis=1)
    return points


def draw_bounding_boxes(line_weight: int, path: str, bounding_boxes: np.ndarray, bb_format="xyxy") -> np.ndarray:
    image = cv2.imread(path)
    width, height, _ = np.shape(image)

    for bb in bounding_boxes:
        points = create_points(bb, bb_format)
        for point in points:
            for i in range(4):
                image = cv2.line(image, point[i % 4], point[(i+1) % 4], (0, 0, 0), line_weight)

    return image
