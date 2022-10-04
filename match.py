import numpy as np
import tensorflow as tf


class Matcher:
    def __init__(
        self,
        model: tf.keras.applications,
        input_preprocessor: tf.keras.applications,
        database: np.ndarray,
        n_neighbours: int,
    ):
        """
        Matcher is class which implements simple matching mechanism for two images based on their embeddings generated
        by neural network. As input, it takes three parameters model, input_preprocessor and database. They will be
        change in next version due allowance to use PyTorch models as well as models from other libraries e.g.
        transformers. # TODO: Add other options for scann.

        :param model: tf.keras.applications models like VGG16 or VGG19.
        :param input_preprocessor: tf.keras.application model input preprocessing function.
        :param database: np.ndarray with all generated embeddings.
        :param n_neighbours: number of potential matches.
        """
        self.model = model(weights="imagenet", include_top=False)
        self.input_preprocessor = input_preprocessor
        self.n_leaves = np.sqrt(database.shape[0])

        self.n_neighbours = n_neighbours

    def align_face(self):
        raise NotImplementedError

    def match_face(self, image: np.ndarray):
        pass

    def get_features(self, image: np.ndarray):
        image = np.expand_dims(image, axis=0)
        image = self.input_preprocessor(image)
        return self.model.predict(image).flatten()
