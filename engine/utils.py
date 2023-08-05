import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def cosine_similarity(query_image_vector: np.ndarray, image_vectors: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between image vectors.

    D is feature vector dimensionality (e.g. 1024)
    N is the number of images in the batch.

    Args:
        query_image_vector: Vectorized image query of (1, D) shape.
        image_vectors: Vectorized images batch of (N, D) shape.

    Returns:
        The vector of (1, N) shape with values in range [-1, 1] where
        1 is max similarity i.e. two vectors are the same.
    """

    dot_product = np.dot(query_image_vector, image_vectors.T)
    query_norm = np.linalg.norm(query_image_vector)
    image_norms = np.linalg.norm(image_vectors, axis=1)

    cosine_similarities = dot_product / (query_norm * image_norms)

    return cosine_similarities


def read_dataset(path: str) -> Tuple[List[np.ndarray], List[str]]:
    path = Path(path)

    if not path.is_dir():
        raise Exception(f"{path} is not a directory")

    if not any(path.iterdir()):
        raise Exception(f"{path} is empty")

    images = []
    image_names = []
    file_count = len(list(path.glob("*.jpg")))

    logging.info(f"Reading {file_count} images from {path}")
    with tqdm(total=file_count) as pbar:
        for image_path in path.glob("*.jpg"):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
                image_names.append(image_path.name)
            pbar.update(1)

    return images, image_names


def create_collage(
    query_image: np.ndarray,
    result: List[Tuple[float, str]],
    dataset: List[np.ndarray],
    image_names: List[str],
) -> None:
    fig, axs = plt.subplots(1, len(result) + 1, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    axs[0].axis("off")
    axs[0].set_title("query image")

    for i, (score, name) in enumerate(result):
        dataset_index = image_names.index(name)
        image = dataset[dataset_index]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[i + 1].imshow(image)
        axs[i + 1].axis("off")
        axs[i + 1].set_title(f"{name}\nSimilarity: {score:.3f}")

    plt.tight_layout()
    plt.savefig("result.png")
