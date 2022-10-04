from abc import ABC, abstractmethod

import os
import wget
import numpy as np

from tqdm import tqdm


class DataSet(ABC):
    def __init__(self, name: str, file_format: str):
        self.dataset_name = name
        self.file_format = file_format

    @abstractmethod
    def download_dataset(self, download_link: str, extension: str):
        pass

    @abstractmethod
    def save_dataset(self, dataset):
        pass

    @abstractmethod
    def generate_embeddings(self, images, n_jobs):
        pass

    @abstractmethod
    def download_images(self, download_path: str, photo_format: str, links, n_jobs):
        pass

    @abstractmethod
    def load_dataset(self):
        pass


class CelebMatcherDataset(DataSet):
    def __init__(self, name="celeb_matcher_dataset", file_format="numpy"):
        """
        Example class implements celebrities dataset which we used as example. It can be used for writing different
        dataset classes and loaders.
        :param name: string with the name of the dataset.
        :param file_format: format of data file to be saved.
        """
        self.dataset_name = name
        self.file_format = file_format
        self.__create_path()

    def __create_path(self):
        os.makedirs(self.dataset_name, exist_ok=True)

    def download_dataset(self, link: str, extension: str):
        wget.download(link, self.dataset_name + extension)

    def save_dataset(self, dataset: np.ndarray):
        if self.file_format == "numpy":
            self.__save_numpy(dataset)

    def __save_numpy(self, dataset: np.ndarray):
        with open(f"{self.dataset_name}.npy", "wb") as fopen:
            np.save(fopen, dataset)

    def generate_embeddings(self, images, n_jobs: int):
        pass

    def download_images(
        self, download_path: str, photo_format: str, links, n_jobs: int
    ):
        # TODO Write it using multithreading or something similar
        for i, link in tqdm(enumerate(links)):
            wget.download(link, f"{download_path}/{i}.{photo_format}")

    def load_dataset(self):
        pass
