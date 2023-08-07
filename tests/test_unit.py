import sys
from pathlib import Path

import cv2
import numpy as np

BASE_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_PATH))

from engine.utils import resize_with_padding
from engine.vectorizer import Vectorizer


class TestUtils:
    def test_resize_with_padding(self):
        test_img = np.zeros((200, 300, 3), dtype=np.uint8)  # (H, W, C)

        resized_img, new_w, new_h, pad_left, pad_top = resize_with_padding(
            test_img, (300, 300), pad_color=255
        )

        assert resized_img.shape == (300, 300, 3)
        assert new_w == 300
        assert new_h == 200
        assert pad_left == 0
        assert pad_top == 50

    def test_vectorizing(self):
        vect = Vectorizer()
        test_img = cv2.imread("tests/samples/car.jpg", cv2.IMREAD_COLOR)

        vectorized_img = vect.transform([test_img])
        test_car_vectorized = np.load("tests/samples/car_vectorized.npy")

        assert np.all(vectorized_img == test_car_vectorized)
