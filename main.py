import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from engine import engine, utils, vectorizer


def cli():
    parser = argparse.ArgumentParser(description="Simple Image Retrieval")
    subparser = parser.add_subparsers(help="commands", dest="commands")

    # Generate sub-commands
    generate = subparser.add_parser("generate", help="Generate feature vector from dataset")
    generate.add_argument("--dataset-path", "-dp", required=True, type=str, help="Dataset path")
    generate.add_argument(
        "--output-dir", "-od", required=True, type=str, help="Feature vector output dir"
    )

    # Search sub-commands
    search = subparser.add_parser("search", help="Image search")
    search.add_argument("--input-image", "-im", required=True, type=str, help="Input image")
    search.add_argument("--dataset-path", "-dp", required=True, type=str, help="Dataset path")
    search.add_argument("--vectors-path", "-vp", required=True, type=str, help="Vectors file path")

    return parser


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = cli().parse_args()

    # Read dataset
    dataset, image_names = utils.read_dataset(args.dataset_path)
    vect = vectorizer.Vectorizer()

    if args.commands == "generate":
        logging.info("Vectorizing dataset...")
        vectorized_dataset = vect.transform(dataset)

        # Save vectors to file
        np.save(Path(args.output_dir) / "vectorized_dataset.npy", vectorized_dataset)
        logging.info("Vectors saved to vectorized_dataset.npy")

    elif args.commands == "search":
        logging.info(f"Loading vectors from {args.vectors_path}...")
        vectorized_dataset = np.load(args.vectors_path)

        # Load input image
        image = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
        if image is None:
            raise Exception(f"{args.input_image} is not a valid image")

        # Search for similar images
        eng = engine.ImageSearchEngine(vect, vectorized_dataset, image_names)
        result = eng.most_similar(image)

        # Create a collage
        collage_file_name = Path(args.input_image).stem + "_result.png"
        utils.create_collage(image, str(collage_file_name), result, dataset, image_names)


if __name__ == "__main__":
    main()
