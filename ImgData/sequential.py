import os
import concurrent.futures
from multiprocessing import cpu_count

import cv2
import numpy as np

from . import Image
from .. import utils
from .. import Operations


class Sequential:
    new_dir: str
    current_dir: str
    operations: list

    def __init__(self, operations: list):
        self.operations = operations
        self.new_dir = ""
        self.current_dir = ""

    def append(self, operations: list):
        for operation in operations:
            self.operations.append(operation)

    def process(self, data: Image,  chunksize: int = 1, max_workers: int = cpu_count()):

        chunksize = utils.get_chunksize(len(data.img_id), max_workers) if chunksize == 1 else chunksize

        self.new_dir = data.directory
        self.current_dir = os.getcwd()
        os.chdir(self.new_dir)

        processed_path = f"{self.new_dir}/processed_images"
        if not os.path.exists(processed_path):
            os.mkdir(processed_path)

        if data.b_boxes is None:
            arguments = np.array([(path, None) for _, path in enumerate(data.img_id)])
        else:
            arguments = np.array([(path, data.b_boxes[i]) for i, path in enumerate(data.img_id)])

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as Executor:
            executed = Executor.map(self.run_sequential, arguments, chunksize=chunksize)

        processed_img = int(np.sum([1 for _ in executed if _]))
        print(f"For {len(data.img_id)} images, {processed_img} images were processed\n")

    def run_sequential(self, inputs: tuple) -> bool:
        img_path = inputs[0]
        bb = inputs[1]

        try:
            img = cv2.imread(f"{self.new_dir}/{img_path}")

            for img_operation in self.operations:
                if img_operation == Operations.Resize:
                    img, bb = img_operation.run(img, bb)
                else:
                    img = img_operation.run(img)

            cv2.imwrite(f"{self.new_dir}/processed_images/{img_path}", img)
            return True

        except ValueError as e:
            print(f"File: {inputs[0]}, Error: {e}")
            return False
