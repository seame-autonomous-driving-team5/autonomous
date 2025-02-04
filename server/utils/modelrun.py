import os
import cv2
import torch
import numpy as np
import traceback
 
import sys

if __name__=="__main__":
    sys.path.append("..")

from flask import Flask, request, jsonify
from torchvision import transforms
from lib.config import cfg
from lib.models import get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils.utils import select_device
from lib.core.postprocess import morphological_process, connect_lane
from lib.utils import show_seg_result

'''
ModelRun class takes the role of running the YOLOPv2 model,  
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

class ModelRun:
    def __init__(self, model_size=(640, 640), resize=False):
        # Model input picture size
        # Размер входного изображения модели
        self.model_size = model_size

        # Load YOLOPv3 model
        # Загружаем модель YOLOPv3
        self.device = select_device(None, "0" if torch.cuda.is_available() else "cpu")  # Use GPU 0 by default
        # Используем GPU 0, если доступен, иначе используем CPU

        self.model = get_net(cfg)  # Получаем модель YOLOPv3
        checkpoint = torch.load("epoch-189.pth", map_location=self.device, weights_only=True)  
        # Загружаем веса модели (замените путь на актуальный)

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device).eval()  # Переводим модель в режим оценки

        # Preprocessing before running YOLOPv3 model
        # Предобработка перед запуском модели YOLOPv3
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Преобразуем изображение в тензор
            transforms.Resize(self.model_size),  # Изменяем размер изображения
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            # Нормализация изображения
        ])

        self.resize = resize  # Флаг изменения размера

    def run(self, img):
        y, x = img.shape[0:2]  # Получаем размеры изображения

        # Pre-processing before running the model
        # Предобработка перед запуском модели
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  
        # Преобразуем изображение в тензор и отправляем на устройство (GPU/CPU)

        # Run the model
        # Запуск модели
        with torch.no_grad():  # Отключаем градиенты для ускорения вычислений
            det_out, da_seg_out, ll_seg_out = self.model(img_tensor)

        # Conducting NMS (Non-Maximum Suppression) after running
        # Применение подавления немаксимумов (NMS) после работы модели
        det_pred = non_max_suppression(det_out[0], conf_thres=0.3, iou_thres=0.45)[0]
        # Применяем NMS с порогами уверенности 0.3 и IoU 0.45

        detections = []  # Список для хранения обнаруженных объектов

        # Saving the data as a dictionary
        # Сохраняем данные в виде словаря
        if det_pred is not None:
            det_pred[:, :4] = scale_coords(img_tensor.shape[2:], det_pred[:, :4], img.shape).round()
            # Приводим координаты к масштабу оригинального изображения
            for *xyxy, conf, cls in det_pred:
                detections.append({
                    "bbox": [int(coord) for coord in xyxy],  # Координаты ограничивающего прямоугольника
                    "confidence": float(conf),  # Уверенность модели
                    "class": int(cls)  # Номер класса
                })

        da_seg_mask, ll_seg_mask = None, None  # Инициализация переменных для сегментации

        if self.resize:
            # Resize segmented data back to the original size
            # Изменение размера сегментированных данных обратно к оригинальному
            da_seg_mask = torch.nn.functional.interpolate(
                da_seg_out[:, :, :, :], size=(y, x), mode='bilinear'
            ).argmax(dim=1).squeeze().cpu().numpy()

            ll_seg_mask = torch.nn.functional.interpolate(
                ll_seg_out[:, :, :, :], size=(y, x), mode='bilinear'
            ).argmax(dim=1).squeeze().cpu().numpy()
        else:
            # Keep segmentation data as is
            # Оставляем сегментированные данные без изменений
            da_seg_mask = da_seg_out.argmax(dim=1).squeeze().cpu().numpy()
            ll_seg_mask = ll_seg_out.argmax(dim=1).squeeze().cpu().numpy()

        # Prepare response
        # Подготавливаем результат
        response = {
            "detections": detections,  # Обнаруженные объекты
            "segmentation": {
                "drivable_area": da_seg_mask.tolist(),  # Область движения
                "lane_lines": ll_seg_mask.tolist()  # Линии разметки
            }
        }

        return response  # Возвращаем результат
