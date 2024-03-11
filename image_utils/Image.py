import os
import numpy as np
from errors import image_errors
import concurrent.futures
from multiprocessing import cpu_count


class Image:
    def __init__(self, current_paths: np.ndarray, new_paths: np.ndarray, bb: np.ndarray, directory: str):
        self.current_paths = current_paths
        self.new_paths = new_paths
        self.b_boxes = bb
        self.directory = directory

    def __add__(self, other):
        self.images = np.concatenate(self.images, other.images, axis=1)
        self.b_boxes = np.concatenate(self.b_boxes, other.b_boxes, axis=1)
        self.new_paths = np.concatenate(self.new_paths, other.new_paths, axis=1)
        if self.directory != other.directory:
            raise image_errors.DirectoryNotSame
        return self

    def __repr__(self):
        if len(self.current_paths >= 5):
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
        self.cpu_count = cpu_count()

    def append(self, operations: list):
        for operation in operations:
            self.operations.append(operation)

    def process(self, data: Image,  chunksize: int = 1, max_workers: int = cpu_count()):

        if not chunksize:
            chunksize = int(max(len(data.current_paths)/100, 1))

        os.chdir(data.directory)
        os.mkdir(f"{data.directory}/processed_images")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as Executor:
            executed = Executor.map(self.run_sequential, data.current_paths, chunksize=chunksize)
            processed = np.sum([1 for _ in executed])
            print(f"For {len(data.current_paths)} images, {processed} images were processed\n")

    def run_sequential(self, data) -> bool:
        try:

            return False
        except Exception as e:
            print(f"There was an error raised in the following file:")
            return True
