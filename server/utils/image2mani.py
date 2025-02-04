import os
import sys

from math import atan2, tanh
import cv2
import numpy as np

if __name__=="__main__":
    sys.path.append("..")

from utils.modelrun import ModelRun
from utils.slidewindow import SlideWindow

'''
Image2Mani class takes the role of suggesting proper steering and speed value.
run() method in Image2Mani class takes the input as an image and the output is a steering value.

Класс Image2Mani отвечает за определение правильных значений рулевого управления и скорости.
Метод run() в классе Image2Mani принимает изображение в качестве входных данных и возвращает значение рулевого управления.

run():: input: image 3D numpy tensor -> output: steer, speed (both are float with range -1~1)

        входные данные: 3D-тензор numpy изображения -> выходные данные: рулевое управление (steer) и скорость (speed) 
        (оба значения float в диапазоне от -1 до 1)
'''

import cv2
import numpy as np
from math import atan2, tanh

class Image2Mani():
    def __init__(self, mode="extreme", speed=0.1):
        print("Model initialized")
        self.modelrun = ModelRun()
        self.slidewindow = SlideWindow()

        # Values required for defining the trapezoidal lane region
        # Значения, необходимые для определения трапециевидной области полосы движения
        self.src_horizon = 330
        self.src_margin_down = 50
        self.src_margin_xaxis = 120

        # Values required for defining the margins in the bird-eye view
        # Значения, необходимые для определения границ в перспективе "птичьего глаза"
        self.dst_margin_up = 50
        self.dst_margin_down = 50

        # Speed setting
        # Установка скорости
        self.speed = speed

        '''There are three modes of determining the steering value:
        - "extreme": Either 0 or 1 (sharp turns).
        - "tanh": Uses the tanh function for smooth control.
        - "normal": Uses an angle-based calculation.

        Существует три режима определения значения рулевого управления:
        - "extreme": Либо 0, либо 1 (резкие повороты).
        - "tanh": Использует функцию tanh для плавного управления.
        - "normal": Использует вычисление угла.
        '''
        if mode not in ["extreme", "normal", "tanh"]:
            raise ValueError("Mode must be 'extreme', 'normal', or 'tanh'. / Вы должны выбрать режим 'extreme', 'normal' или 'tanh'.")
        self.mode = mode

        # Threshold determining the car's sensitivity to curves
        # Порог чувствительности автомобиля к кривизне дороги (чем меньше значение, тем выше чувствительность)
        self.threshold = 50

    # Convert binary image values (0-1) to grayscale image values (0-255)
    # Преобразование двоичного изображения (0-1) в градации серого (0-255)
    def binary2img(self, bin):
        img = cv2.normalize(bin, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = img.astype(np.uint8)
        return img

    # Convert grayscale image values (0-255) back to binary (0-1)
    # Преобразование градаций серого (0-255) обратно в двоичный формат (0-1)
    def img2binary(self, img):
        bin = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return bin

    # Convert image to bird-eye's view
    # Преобразование изображения в вид "птичьего глаза"
    def bird_eyes_view(self, img):
        img = self.binary2img(img)  # Convert binary to grayscale for visualization
        # Преобразуем двоичное изображение в градации серого для визуализации

        y, x = img.shape[0:2]  # Get image dimensions
        # Получаем размеры изображения

        # Define the trapezoidal region for lane detection
        # Определение трапециевидной области для обнаружения полосы движения
        src = np.float32([
            [self.src_margin_xaxis, self.src_horizon],  # Left-top / Левый верх
            [0, y - self.src_margin_down],  # Left-bottom / Левый низ
            [x - self.src_margin_xaxis, self.src_horizon],  # Right-top / Правый верх
            [x, y - self.src_margin_down]  # Right-bottom / Правый низ
        ])

        # Define the transformed region in bird-eye's view
        # Определение области в перспективе "птичьего глаза"
        dst = np.float32([
            [self.dst_margin_up, 0],  # Left-top / Левый верх
            [self.dst_margin_down, y],  # Left-bottom / Левый низ
            [x - self.dst_margin_up, 0],  # Right-top / Правый верх
            [x - self.dst_margin_down, y]  # Right-bottom / Правый низ
        ])

        # Warp the image based on the defined transformation
        # Преобразование изображения на основе заданного преобразования
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(img, matrix, (x, y))
        return warped_img

    # Determine the steering value based on lane curvature
    # Определение значения рулевого управления в зависимости от кривизны полосы
    def determine_steer(self, x_location, img):
        y, x = img.shape[0:2]
        cte = x_location - x // 2  # Cross-track error (偏差)

        '''Three modes of determining steering value:
        - "extreme": Either 0 or 1 based on threshold.
        - "tanh": Uses a hyperbolic tangent function.
        - "normal": Uses an angle-based approach.

        Три режима определения значения рулевого управления:
        - "extreme": Либо 0, либо 1 в зависимости от порога.
        - "tanh": Использует гиперболический тангенс.
        - "normal": Основан на угловом вычислении.
        '''
        if self.mode == "extreme":
            if -self.threshold <= cte <= self.threshold:
                steer = 0
            else:
                steer = 1 if cte >= self.threshold else -1

        elif self.mode == "tanh":
            steer = tanh(cte / self.threshold)

        else:
            steer = atan2(cte, y - self.threshold) / (np.pi / 2)

        return steer

    def run(self, img):
        # Run the YOLOPv2 model first
        # Запускаем модель YOLOPv2
        response = self.modelrun.run(img)

        # Convert drivable area and lane line segmentation to bird-eye's view
        # Преобразуем область движения и разметку полос в вид "птичьего глаза"
        da_birdeye = self.bird_eyes_view(np.array(response["segmentation"]["drivable_area"]))
        ll_birdeye = self.bird_eyes_view(np.array(response["segmentation"]["lane_lines"]))

        # Get the x-location of the lane center using a sliding window
        # Определяем положение центра полосы с помощью скользящего окна
        slided_img, x_location = self.slidewindow.slideWindow(ll_birdeye)

        # Determine the steering value based on lane curvature
        # Определяем значение рулевого управления на основе кривизны полосы
        steer = self.determine_steer(x_location, img)

        return steer, self.speed  # Возвращаем значения рулевого управления и скорости


if __name__ == "__main__":
    img = cv2.imread("../labimage.jpeg")
    modelrun = Image2Mani(mode="tanh")
    steer, speed = modelrun.run(img)
    print("steer:", steer, "speed:", speed)
    cv2.waitKey()
