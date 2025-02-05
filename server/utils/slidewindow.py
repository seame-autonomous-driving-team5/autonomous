import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import *
from matplotlib.pyplot import *
import math
import random
 
TOTAL_CNT = 50

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 128, 255)
PURPLE = (255, 0, 255)

if __name__=="__main__":
    sys.path.append("..")

'''
WindowRange is the class representing a specific rectangular space in the image,  
which will be called "Window".

WindowRange — это класс, представляющий определенную прямоугольную область на изображении,  
которая будет называться "Окно" (Window).
'''

import random
import cv2

class WindowRange:
    def __init__(self, top, bottom, left, right, color=None):
        # Top, bottom, left, and right positions of the window on the image
        # Верхняя, нижняя, левая и правая границы окна на изображении
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

        # The color of the window when drawn. If not set, a random color is chosen.
        # Цвет окна при его отображении. Если не задан, выбирается случайный цвет.
        random_color = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
        self.color = random_color[random.randrange(len(random_color))] if color is None else color
    
    # Given x and y coordinates, this method returns the indexes of points inside the window.
    # Получая координаты x и y, этот метод возвращает индексы точек, находящихся внутри окна.
    def indexes_in_range_by_xy(self, xs, ys):
        if len(xs) != len(ys):
            raise ValueError("The number of x-coordinates must match the number of y-coordinates")
            # Количество x-координат должно соответствовать количеству y-координат
        
        return ((xs >= self.left) & (xs <= self.right) & 
                (ys <= self.top) & (ys > self.bottom)).nonzero()[0]
    
    # Draws the window rectangle on the image.
    # Отображает границы окна на изображении.
    def draw(self, img):
        cv2.rectangle(img, (self.left, self.bottom), (self.right, self.top), self.color, 1)  
  

'''
SlideWindow class helps the computer to detect where the lane is, 
by sliding the fixed-size window. 
It determines where the lane line is, 
and give a clue how much the lane is curved.

slidewindow() of slided window takes the input as image and
the output as image with slided window, 
and x coordinate which is essential clue to know how curved the lane.
'''

'''Класс SlideWindow помогает компьютеру определить, где находится полоса движения, используя метод скользящего окна фиксированного размера.  
Скользящее окно определяет местоположение линии разметки полосы движения, а также помогает оценить, насколько полоса изогнута.  

Метод slideWindow() принимает изображение в качестве входных данных  
и возвращает изображение с отмеченными скользящими окнами, а также x-координату положения полосы движения.'''


class SlideWindow:
    
    def __init__(self, left_lane_l=50, left_lane_r=280, 
                 right_lane_l=350, right_lane_r=560, 
                 bottom=90, top=500, 
                 circle_height=120):
        
        # Define the assumed range for left and right lane detection
        # Определяем предполагаемый диапазон для обнаружения левой и правой полосы
        self.left_lane_range = WindowRange(top, bottom, left_lane_l, left_lane_r, color=YELLOW)
        self.right_lane_range = WindowRange(top, bottom, right_lane_l, right_lane_r, color=PURPLE)

        # Set sliding window dimensions
        # Устанавливаем размеры скользящего окна
        self.wd_width = 80
        self.wd_height = 20

        # Stride determines how much the window moves in each iteration
        # Шаг определяет, насколько окно перемещается на каждой итерации
        self.stride = 20

        # Define additional parameters
        # Определяем дополнительные параметры
        self.circle_height = circle_height
        self.road_width = 0.55

        # Minimum number of pixels required for lane detection
        # Минимальное количество пикселей, необходимое для обнаружения полосы движения
        self.window_minpix = 100
        self.lane_minpix = 300

    # Ensure x value is within image bounds
    # Гарантируем, что значение x находится в пределах изображения
    def xrange(self, x_val, img):
        y, x = img.shape[0:2]
        return max(0, min(x, x_val))
    
    # Ensure y value is within image bounds
    # Гарантируем, что значение y находится в пределах изображения
    def yrange(self, y_val, img):
        y, x = img.shape[0:2]
        return max(0, min(y, y_val))
    
    # Adjust position inside the bounding box based on pixel distribution
    # Настраиваем положение внутри ограничивающего окна на основе распределения пикселей
    def position_setting_inside_box(self, windowRange, ys, xs, origin_val):
        
        indexes_in_range = windowRange.indexes_in_range_by_xy(xs, ys)

        # If enough pixels are found, compute the mean x position
        # Если найдено достаточно пикселей, вычисляем среднее значение x
        if len(indexes_in_range) > self.window_minpix:
            return int(np.mean(xs[indexes_in_range]))
        
        # If a few pixels are found, fit a quadratic curve and estimate position
        # Если найдено несколько пикселей, подгоняем квадратичную кривую и оцениваем положение
        elif len(indexes_in_range) > 2:
            p = np.polyfit(ys[indexes_in_range], xs[indexes_in_range], 2)
            return int(np.polyval(p, windowRange.top))
        
        # Otherwise, retain original value
        # В противном случае сохраняем исходное значение
        else:
            return origin_val
        
    # Draw lane range on image
    # Рисуем область полосы движения на изображении
    def draw_range(self, img, indexes, xs, ys, color):
        for idx in indexes:
            img = cv2.circle(img, (xs[indexes[idx]], ys[indexes[idx]]), 1, color, -1)

    # Draw a circle at a specific x-location on the image
    # Рисуем круг в определенной позиции x на изображении
    def draw_x_location(self, x_location, img, color):
        cv2.circle(img, (x_location, self.circle_height), 10, color, -1)
        
    # Main sliding window function to detect lane positions
    # Основная функция скользящего окна для обнаружения положения полосы движения
    def slideWindow(self, img):
        y, x = img.shape[0:2]
        out_img = np.dstack((img, img, img)) * 255  # Convert grayscale to RGB
        # Преобразуем изображение в RGB
        x_location = x // 2  # Initialize vehicle position at image center
        # Инициализируем положение автомобиля в центре изображения

        # Draw the left and right lane search regions
        # Рисуем области поиска левой и правой полосы
        self.left_lane_range.draw(img)
        self.right_lane_range.draw(img)

        # Identify all nonzero pixels in the image
        # Определяем все ненулевые пиксели на изображении
        nonzero = img.nonzero()
        nonzeroy, nonzerox = nonzero[0], nonzero[1]
        left_lane_indexes = self.left_lane_range.indexes_in_range_by_xy(nonzerox, nonzeroy)
        right_lane_indexes = self.right_lane_range.indexes_in_range_by_xy(nonzerox, nonzeroy)

        # Case 1: Both lanes detected
        # Случай 1: Обнаружены обе полосы движения
        if len(left_lane_indexes) > self.lane_minpix and len(right_lane_indexes) > self.lane_minpix:
            x_avg_left = int(np.mean(nonzerox[left_lane_indexes]))
            y_top_left = int(np.max(nonzeroy[left_lane_indexes]))

            x_avg_right = int(np.mean(nonzerox[right_lane_indexes]))
            y_top_right = int(np.max(nonzeroy[right_lane_indexes]))

            # Slide window from top to bottom
            for top in range(max(y_top_left, y_top_right), 0, -self.stride):
                
                # Create sliding windows for left and right lanes
                left_window = WindowRange(top=top,
                                          bottom=self.yrange(top - self.wd_height, img),
                                          left=x_avg_left - (self.wd_width // 2),
                                          right=x_avg_left + (self.wd_width // 2),
                                          color=BLUE)
                
                right_window = WindowRange(top=top,
                                          bottom=self.yrange(top - self.wd_height, img),
                                          left=x_avg_right - (self.wd_width // 2),
                                          right=x_avg_right + (self.wd_width // 2),
                                          color=GREEN)
                
                # Draw windows on image
                left_window.draw(img)
                right_window.draw(img)

                # Update lane positions
                x_avg_left = self.position_setting_inside_box(left_window, nonzeroy, nonzerox, x_avg_left)
                x_avg_right = self.position_setting_inside_box(right_window, nonzeroy, nonzerox, x_avg_right)

                # Estimate x location of the vehicle
                if (self.circle_height - self.stride//2 <= left_window.top <= self.circle_height + self.stride//2):
                    x_location = self.xrange((x_avg_left + x_avg_right) // 2, img)
        
        # Case 2: Only left lane detected
        # Случай 2: Обнаружена только левая полоса
        elif len(left_lane_indexes) > self.lane_minpix and len(right_lane_indexes) < self.lane_minpix:
            # Same logic as above but only for left lane

            x_avg_left = int(np.mean(nonzerox[left_lane_indexes]))
            y_top_left = int(np.max(nonzeroy[left_lane_indexes]))

            for top in range(y_top_left, 0, -self.stride):

                left_window = WindowRange(top = top,
                                          bottom = self.yrange(top - self.wd_height, img),
                                          left = x_avg_left - (self.wd_width // 2) ,
                                          right = x_avg_left + (self.wd_width // 2),
                                          color = BLUE )
                
                right_window = WindowRange(top = top,
                                           bottom = self.yrange(top - self.wd_height, img),
                                           left = x_avg_left - (self.wd_width // 2) + int(x * self.road_width),
                                           right = x_avg_left + (self.wd_width // 2) + int(x * self.road_width),
                                           color = GREEN )
                
                left_window.draw(img)
                right_window.draw(img)

                x_avg_left = self.position_setting_inside_box(left_window, nonzeroy, nonzerox, x_avg_left)

                if (self.circle_height - self.stride//2 <= left_window.top <= self.circle_height + self.stride//2):
                    x_location = self.xrange(x_avg_left + int(x * self.road_width //2))
        
        # Case 3: Only right lane detected
        # # Случай 3: Обнаружена только правая полоса
        elif len(left_lane_indexes) < self.lane_minpix and len(right_lane_indexes) > self.lane_minpix:
            # Same logic as above but only for right lane

            x_avg_right = int(np.mean(nonzerox[right_lane_indexes]))
            y_top_right = int(np.max(nonzeroy[right_lane_indexes]))

            for top in range(y_top_right, 0, -self.stride):
                left_window = WindowRange(top = top,
                                          bottom = self.yrange(top - self.wd_height, img),
                                          left = x_avg_right - (self.wd_width // 2) - int(x * self.road_width),
                                          right = x_avg_right + (self.wd_width // 2) - int(x * self.road_width),
                                          color = BLUE )
                
                right_window = WindowRange(top = top,
                                           bottom = self.yrange(top - self.wd_height, img),
                                           left = x_avg_right - (self.wd_width // 2),
                                           right = x_avg_right + (self.wd_width // 2),
                                           color = GREEN )
                
                left_window.draw(img)
                right_window.draw(img)

                y_avg_right = self.position_setting_inside_box(right_window, nonzeroy, nonzerox, y_avg_right)

                if (self.circle_height - self.stride//2 <= right_window.top <= self.circle_height + self.stride//2):
                    x_location = self.xrange(x_avg_right - int(x * self.road_width //2))

        else:
            pass

        # Draw vehicle's estimated position on the image
        # Рисуем предполагаемое положение автомобиля на изображении
        self.draw_x_location(x_location, img, RED)
        return img, x_location


if __name__ == "__main__":
    img = cv2.imread("birdeye.png")
    sdw = SlideWindow()
    img, x_location = sdw.slideWindow(img)
    print("x_location: " , x_location)
    cv2.imshow("output", img)
    cv2.waitKey()