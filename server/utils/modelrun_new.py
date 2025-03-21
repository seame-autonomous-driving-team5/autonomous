'''
ModelRun class takes the role of running the YOLOPv2 model,
including pre-processing and post-processing,
and run() in ModelRun takes the output as detected objects,  
segmented driving area, and segmented lane lines area.

Класс ModelRun выполняет модель YOLOPv2,  
а метод run() в ModelRun возвращает обнаруженные объекты,  
сегментированную зону для движения и сегментированные линии разметки.

run():: input: image 3D numpy tensor ->  
        output: dict object {"detections": detected objects, "segmentation": {"drivable_area": , "lane_line": segmented lane line}}

        входные данные: 3D тензор numpy изображения ->  
        выходные данные: объект dict {"detections": обнаруженные объекты, "segmentation": {"drivable_area": , "lane_line": сегментированные линии разметки}}

        detected objects (обнаруженные объекты) - это список словарей, каждый элемент имеет структуру:  
        {"bbox": (left, top, right, bottom), "confidence": float , "class": int class number}  
        - "bbox" - координаты ограничивающего прямоугольника (левая, верхняя, правая, нижняя границы)  
        - "confidence" - уверенность модели (float)  
        - "class" - номер класса объекта (int)  

        drivable area (зона движения) - это тензор (H*W), где элемент равен 1, если проезжаемая зона, и 0 в противном случае.  
        lane line (линии разметки) - также тензор (H*W), где элемент равен 1, если это линия, и 0 в противном случае.  
'''

import torch
import cv2
import random
from torchvision import transforms
from ultralytics import YOLO

class ModelRun:
    def __init__(self, model_size=(640, 640), resize=False):
        self.model = YOLO("best.engine")


    def postprocess(self, results):
        result = results[0].masks.data[0]
        result = result.cpu().numpy()

        return {"lane": result[0], "stop_line": result[1]}


    def run(self, img):
        results = self.model(img)
        result = self.postprocess(results)

        return result  # Возвращаем результат
