import os
import numpy as np
import concurrent.futures
from multiprocessing import cpu_count
import cv2
import Operations
from errors import image_errors


class Image:
    def __init__(self, current_paths: np.ndarray, directory: str, bb: np.ndarray | None = None):
        self.current_paths = current_paths
        self.b_boxes = bb
        self.directory = directory

    def __add__(self, other):
        self.images = np.concatenate(self.images, other.images, axis=1)
        self.b_boxes = np.concatenate(self.b_boxes, other.b_boxes, axis=1)

        if self.directory != other.directory:
            raise image_errors.DirectoryNotSame

        return self

    def __repr__(self):
        if len(self.current_paths)>= 5:
            st = f"Number of images:{len(self.current_paths)}\nThe first 5 paths are:"
            for i in range(5):
                st = st + f"\n{self.current_paths[i]}"
        else:
            st = f"Number of images:{len(self.current_paths)}\nThe first few paths are:"
            for path in self.current_paths:
                st = st + f"\n{path}"
        return st


class Sequential:
    def __init__(self, operations: list):
        self.operations = operations

    def append(self, operations: list):
        for operation in operations:
            self.operations.append(operation)

    def process(self, data: Image,  chunksize: int = 1, max_workers: int = cpu_count()):

        if chunksize == 1:
            chunksize = int(max(len(data.current_paths)/100, 1))

        os.chdir(data.directory)
        os.mkdir(f"{data.directory}/processed_images")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as Executor:
            executed = Executor.map(self.run_sequential, data.current_paths, chunksize=chunksize)
            processed_img = np.sum([1 for _ in executed])

            print(f"For {len(data.current_paths)} images, {processed_img} images were processed\n")

    def run_sequential(self, img_path, bb: np.ndarray | None) -> bool:
        try:
            cwd = os.getcwd()

            img = cv2.imread(f"{cwd}/{img_path}")

            for img_operation in self.operations:

                if img_operation == Operations.Resize:
                    img, bb = img_operation.run(img, bb)
                else:
                    img = img_operation.run(img)

            image_processed = cv2.imwrite(f"{cwd}/processed_images/{img_path}", img)

            return image_processed
        except Exception as e:
            print(f"There was an error raised in the following file: {img_path}, with the following error: {e}")
            return False
