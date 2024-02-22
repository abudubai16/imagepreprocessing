import asyncio
import cv2
import numpy as np


async def read_path(path: str) -> np.ndarray:
    img = await cv2.imread(path)
    return img


async def read_paths(paths: np.ndarray) -> np.ndarray:
    images = await asyncio.gather(*[read_path(path) for path in paths])
    return images


async def write_path(image: np.ndarray, path: str):
    cv2.imwrite(path, image)


async def write_paths(images: np.ndarray, paths: np.ndarray):
    tasks = [asyncio.create_task(write_path(image=images[i], path=paths[i])) for i in range(len(images))]
    tasks = [await task for task in tasks]
