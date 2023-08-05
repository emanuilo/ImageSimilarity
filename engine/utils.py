import numpy as np


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
