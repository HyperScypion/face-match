import os
import tensorflow as tf

from abc import ABC
from tqdm import tqdm
from pathlib import Path
from face_preprocessor import SimplePreprocessor


class SimpleFaceEmbedder(ABC):
    """
    SimpleFaceEmbedder class implements usage of embedder model.
    """
    def __init__(self, images_path: Path, model: Path, pretrained: bool = False):
        """
        :param images_path: Path to images directory.
        :param model:
        :param pretrained: Used for using pretrained model from tensorflow.
        """
        self.images_path = images_path
        self.model = model
        self.pretrained = pretrained

    def load_model(self):
        if self.pretrained is True:
            model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        else:
            model = tf.keras.models.load_model(self.model)
        return model

    def generate_embeddings(self):
        model = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        preprocessor = SimplePreprocessor(self.images_path)
        faces = preprocessor.preprocess((224, 224))
        preprocessed_faces = [tf.keras.applications.vgg16.preprocess_input(face) for face in tqdm(faces)]
        import numpy as np
        embeddings = model.predict(np.array(preprocessed_faces)).flatten()
        return embeddings

    def __check_support(self):
        pass

