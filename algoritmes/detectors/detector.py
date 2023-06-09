import cv2

from algoritmes.detectors.yolov5 import YOLOv5Detector
from algoritmes.detectors.yolov6 import YOLOv6Detector
from algoritmes.detectors.yolov7 import YOLOv7Detector
from algoritmes.detectors.yolor import YOLOrDetector
from algoritmes.detectors.yolox import YOLOxDetector
from algoritmes.detectors.easyocr_detector import TextDetector

from algoritmes.detectors.utils.weights_path import get_weight_path
from algoritmes.detectors.utils.cfg_path import get_cfg_path
from algoritmes.detectors.utils.exp_name import get_exp__name
from .yolov8 import YOLOv8Detector


class Detector:
    def __init__(self,
                 model_flag: int,
                 weights: str = None,
                 use_cuda: bool = True,
                 recognizer:int = None):
        
        self.model = self._select_detector(model_flag, weights, use_cuda, recognizer)
    def _select_detector(self, model_flag, weights, cuda, recognizer):
        # Get required weight using model_flag
        if weights and weights.split('.')[-1] == 'onnx':
            onnx = True
            weight = weights
        elif weights:
            onnx = False
            weight = weights
        else:
            onnx, weight = get_weight_path(model_flag)

        if model_flag in range(0, 20):
            _detector = YOLOv5Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        elif model_flag in range(20, 34):
            _detector = YOLOv6Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        elif model_flag in range(34, 48):
            _detector = YOLOv7Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        elif model_flag in range(48, 58):
            # Get Configuration file for Yolor
            if model_flag in range(48, 57, 2):
                cfg = get_cfg_path(model_flag)
            else:
                cfg = None
            _detector = YOLOrDetector(weights=weight,
                                      cfg=cfg,
                                      use_onnx=onnx,
                                      use_cuda=cuda)

        elif model_flag in range(58, 72):
            # Получить exp-файл и соответствующую модель только для pytorch
            if model_flag in range(58, 71, 2):
                exp, model_name = get_exp__name(model_flag)
            else:
                exp = model_name = None
            _detector = YOLOxDetector(model_name=model_name,
                                      exp_file=exp,
                                      weights=weight,
                                      use_onnx=onnx,
                                      use_cuda=cuda)
        elif model_flag in range(72, 82):
            # Получить exp-файл и соответствующую модель только для pytorch
            _detector = YOLOv8Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        # Get TextDetector model
        elif model_flag  in range(82, 85):
            _detector = TextDetector(detect_network=weight, use_cuda=cuda)
        return _detector

    def get_detector(self):
        return self.model

    def detect(self,
               image: list,
               **kwargs: dict):
        return self.model.detect(image, **kwargs)
