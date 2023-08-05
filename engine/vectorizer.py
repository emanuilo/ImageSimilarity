from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


class Vectorizer:
    def __init__(self) -> None:
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.model = torch.nn.Sequential(OrderedDict([*(list(self.model.named_children())[:-1])]))

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def transform(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Transform list of images into numpy vectors of image features.

        Images should be preprocessed first (padding, resize, normalize,..)

        Args:
            images: The sequence of raw images.

        Returns:
            Vectorized images as numpy array of (N, D) shape where
            N is the number of images, and D is feature vector
            size after running it through the vectorizer.
        """

        vectors = []

        for image in tqdm(images):
            preprocessed_img = self._preprocess(image)
            vector = self.model(preprocessed_img)
            vector = vector.detach().numpy()
            vector = np.reshape(vector, (1, -1))
            vectors.append(vector)

        output = np.concatenate(vectors, axis=0)
        return output

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image before vectorization.

        Args:
            image: The raw image.

        Returns:
            The preprocessed image.
        """
        image = self.transforms(image)
        image = torch.unsqueeze(image, 0)
        return image
