import os
import cv2
import tensorflow as tf

from abc import ABC, abstractmethod
from face_align import *

from pathlib import Path
from tqdm import tqdm


class SimplePreprocessor:
    def __init__(self, images_path: Path):
        self.images_path = images_path
        # self.face_aligner = FaceAligner(predictor_backend="dlib")

    def preprocess(self, image_size: tuple):
        faces = []
        for image in tqdm(os.listdir(self.images_path)):
            print(self.images_path + image)
            image = cv2.imread(self.images_path + image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size)
            print(image.shape)
            # image = self.face_aligner.align(image)
            faces.append(image)
        return faces
