import copy
import cv2
from loguru import logger
import os
import time
import algoritmes.utils as utils
from algoritmes.trackers import Tracker
from algoritmes.detectors import Detector
from algoritmes.recognizers import TextRecognizer
from algoritmes.utils.default_cfg import config
import numpy as np


class TRAOne:
    def __init__(self,
                 detector: int = 0,
                 tracker: int = -1,
                 weights: str = None,
                 use_cuda: bool = True,
                 recognizer: int = None,
                 languages: list = ['en']
                 ) -> None:

        self.use_cuda = use_cuda

        # get detector object
        self.detector = self.get_detector(detector, weights, recognizer)
        self.recognizer = self.get_recognizer(recognizer, languages=languages)
    
        if tracker == -1:
            self.tracker = None
            return
            
        self.tracker = self.get_tracker(tracker)

    def get_detector(self, detector: int, weights: str, recognizer):
        detector = Detector(detector, weights=weights,
                            use_cuda=self.use_cuda, recognizer=recognizer).get_detector()
        return detector

    def get_recognizer(self, recognizer: int, languages):
        if recognizer == None:
            return None
        recognizer = TextRecognizer(recognizer,
                            use_cuda=self.use_cuda, languages=languages).get_recognizer()

        return recognizer

    def get_tracker(self, tracker: int):
        tracker = Tracker(tracker, self.detector,
                          use_cuda=self.use_cuda)
        return tracker

    def _update_args(self, kwargs):
        for key, value in kwargs.items():
            if key in config.keys():
                config[key] = value
            else:
                print(f'"{key}" аргумент не найден! допустимые аргументы: {list(config.keys())}')
                exit()
        return config

    def track_stream(self,
                    stream_url,
                    **kwargs
                    ):

        output_filename = 'result.mp4'
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(stream_url, config):
            # возвращаем bbox_details, frame_details в основной скрипт
            yield bbox_details, frame_details

    def track_video(self,
                    video_path,
                    **kwargs
                    ):            
        output_filename = os.path.basename(video_path)
        kwargs['filename'] = output_filename
        config = self._update_args(kwargs)

        for (bbox_details, frame_details) in self._start_tracking(video_path, config):
            # yeild bbox_details, frame_details to main script
            yield bbox_details, frame_details

    def detect(self, source, **kwargs)->np.ndarray:
        """Функция для выполнения обнаружения в img

            Аргументы:
            источник (_type_): если str прочитал изображение. если nd.array передает его непосредственно для обнаружения

            Возвращается:
        _type_: ndarray обнаружения
        """
        if isinstance(source, str):
            source = cv2.imread(source)
        return self.detector.detect(source, **kwargs)
    
    def detect_text(self, image):
        horizontal_list, _ = self.detector.detect(image)
        if self.recognizer is None:
                raise TypeError("Распознаватель не может быть ни одним")
            
        return self.recognizer.recognize(image, horizontal_list=horizontal_list,
                            free_list=[])

    def track_webcam(self,
                     cam_id=0,
                     **kwargs):
        output_filename = 'results.mp4'

        kwargs['filename'] = output_filename
        kwargs['fps'] = 29
        config = self._update_args(kwargs)


        for (bbox_details, frame_details) in self._start_tracking(cam_id, config):
            # возвращаем bbox_details, frame_details в основной скрипт
            yield bbox_details, frame_details
        
    def _start_tracking(self,
                        stream_path: str,
                        config: dict) -> tuple:

        if not self.tracker:
            print(f'Трекер не выбран. используйте функцию detect() для выполнения обнаружения или передачи трекера')
            exit()

        fps = config.pop('fps')
        output_dir = config.pop('output_dir')
        filename = config.pop('filename')
        save_result = config.pop('save_result')
        display = config.pop('display')
        draw_trails = config.pop('draw_trails')
        class_names = config.pop('class_names')

        cap = cv2.VideoCapture(stream_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
            logger.info(f"путь к сохранению видео - это {save_path}")

            video_writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (int(width), int(height)),
            )

        frame_id = 1
        tic = time.time()

        prevTime = 0

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break
            im0 = copy.deepcopy(frame)

            bboxes_xyxy, ids, scores, class_ids = self.tracker.detect_and_track(
                frame, config)
            elapsed_time = time.time() - start_time

            logger.info(
                'frame {}/{} ({:.2f} ms)'.format(frame_id, int(frame_count),
                                                 elapsed_time * 1000))

            if self.recognizer:
                res = self.recognizer.recognize(im0, horizontal_list=bboxes_xyxy,
                            free_list=[])
                im0 = utils.draw_text(im0, res)
            else:
                im0 = utils.draw_boxes(im0,
                                    bboxes_xyxy,
                                    class_ids,
                                    identities=ids,
                                    draw_trails=draw_trails,
                                    class_names=class_names)

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.line(im0, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                        225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            if display:
                cv2.imshow(' Образец', im0)
            if save_result:
                video_writer.write(im0)

            frame_id += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # выдать требуемые значения в виде (bbox_details, frames_details)
            yield (bboxes_xyxy, ids, scores, class_ids), (im0 if display else frame, frame_id-1, fps)

        tac = time.time()
        print(f'Общее затраченное время: {tac - tic:.2f}')


if __name__ == '__main__':
    # asone = ASOne(tracker='norfair')
    # asone = ASOne()
    #
    # asone.start_tracking('data/sample_videos/video2.mp4',
    #                      save_result=True, display=False)
