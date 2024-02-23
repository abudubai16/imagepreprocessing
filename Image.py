import asyncio
import os
import numpy as np
import utility_functions as func


class Image:
    current_paths: np.ndarray
    new_paths: np.ndarray
    b_boxes: np.ndarray
    directory: str

    def __init__(self, current_paths: np.ndarray, new_paths: np.ndarray, bb: np.ndarray, directory: str):
        self.current_paths = current_paths
        self.new_paths = new_paths
        self.b_boxes = bb
        self.directory = directory

    def __add__(self, other):
        self.images = np.concatenate(self.images, other.images, axis=1)
        self.b_boxes = np.concatenate(self.b_boxes, other.b_boxes, axis=1)
        self.new_paths = np.concatenate(self.new_paths, other.new_paths, axis=1)
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
    operations: list
    images: np.ndarray

    def __init__(self, operations: list):
        self.operations = operations

    def append(self, operations: list):
        for operation in operations:
            self.operations.append(operation)

    async def run_process(self, images: np.ndarray) -> tuple:
        images = await asyncio.gather(*[self.sequential(image) for image in images])
        return images

    def process(self, data: Image):
        images = asyncio.run(func.read_paths(data.current_paths))
        images = asyncio.run(self.run_process(images=images))
        os.chdir(f"{data.directory}")
        asyncio.run(func.write_paths(images, data.new_paths))

    async def sequential(self, image: np.ndarray) -> np.ndarrray:
        for operation in self.operations:
            image = operation[0](image=image, *operation[1])
        return image
