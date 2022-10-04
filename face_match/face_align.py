import dlib
import mediapipe as mp

from tqdm import tqdm


class FaceAligner:
    """
    Implementation based on https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    """
    def __init__(self, predictor_backend: str):
        self.predictor_backend = predictor_backend
        self.face_detector = self.__set_predictor(predictor_backend)

    @staticmethod
    def __set_predictor(predictor_backend: str):
        if predictor_backend == "mediapipe":
            return mp.solutions.face_detection
        elif predictor_backend == "dlib":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def find_faces(self, images, n_faces: int):
        if self.predictor_backend == "mediapipe":
            self.run_mediapipe(images, n_faces)

    def run_mediapipe(self, images, n_faces: int, **kwargs):
        """
        # TODO add removing faces
        :param images:
        :param n_faces:
        :param kwargs:
        :return:
        """
        with self.face_detector.FaceDetection(**kwargs) as face_detection:
            for image in tqdm(images):
                # image needs to be opencv RGB image
                results = face_detection.process(image)

                if not results.detections:
                    continue

                print(results.detections)
                raise NotImplementedError

    def align(self):
        raise NotImplementedError

