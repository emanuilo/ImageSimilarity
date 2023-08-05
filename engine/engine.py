from typing import List, Tuple

import numpy as np

from engine.utils import cosine_similarity
from engine.vectorizer import Vectorizer


class ImageSearchEngine:
    def __init__(
        self, vectorizer: Vectorizer, vectorized_dataset: np.ndarray, image_names: List[str]
    ) -> None:
        self.vectorizer = vectorizer
        self.vectorized_dataset = vectorized_dataset
        self.image_names = image_names

    def most_similar(self, query: np.ndarray, n: int = 5) -> List[Tuple[float, str]]:
        """Return top n most similar images from corpus.

        Input image should be cleaned and vectorized with fitted
        Vectorizer to get query image vector. After that, use
        the cosine_similarity function to get the top n most similar images
        from the data set.

        Args:
            query: The raw query image input from the user.
            n: The number of similar image names returned from the corpus.

        Returns:
            The list of top n most similar images from the corpus along
            with similarity scores. Note that returned str is image name.
        """
        query_img_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_img_vector, self.vectorized_dataset)
        cosine_similarities = cosine_similarities.squeeze()
        most_similar_ids = np.argsort(cosine_similarities)[::-1][:n]

        most_similar_imgs = [
            (cosine_similarities[i], self.image_names[i]) for i in most_similar_ids
        ]

        return most_similar_imgs
