import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from engine import engine, vectorizer


def cli():
    parser = argparse.ArgumentParser(description="Simple Image Retrieval")
    parser.add_argument("--input-image", "-im", required=True, type=str, help="Input image")
    parser.add_argument("--dataset-path", "-dp", required=True, type=str, help="Dataset path")
    parser.add_argument(
        "--load-vectors", "-lv", required=False, action="store_true", help="Load vectors from file"
    )
    parser.add_argument(
        "--vectors-path", "-vp", required=False, type=str, help="Vectors file path"
    )

    return parser


def read_dataset(path: str):
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
):
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


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = cli().parse_args()

    # Read dataset
    dataset, image_names = read_dataset(args.dataset_path)

    # Vectorize dataset
    vect = vectorizer.Vectorizer()
    if args.load_vectors:
        logging.info(f"Loading vectors from {args.vectors_path}...")
        vectorized_dataset = np.load(args.vectors_path)
    else:
        logging.info("Vectorizing dataset...")
        vectorized_dataset = vect.transform(dataset)
        np.save("vectorized_dataset.npy", vectorized_dataset)
        logging.info("Vectors saved to vectorized_dataset.npy")

    # Load input image
    image = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception(f"{args.input_image} is not a valid image")

    # Search for similar images
    eng = engine.ImageSearchEngine(vect, vectorized_dataset, image_names)
    result = eng.most_similar(image)

    # Create a collage
    create_collage(image, result, dataset, image_names)


if __name__ == "__main__":
    main()
