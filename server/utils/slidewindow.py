import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import *
from matplotlib.pyplot import *
import math
import random
# float로 조향값 public
 
TOTAL_CNT = 50

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 128, 255)
PURPLE = (255, 0, 255)


class WindowRange:
    def __init__(self, top, bottom, left, right, color = None):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

        random_color = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
        self.color = random_color[random.randrange(len(random_color))] if color == None else color
    
    def indexes_in_range_by_xy(self, xs, ys):
        if len(xs) != len(ys):
            raise ValueError("x coordinates of y coordinates must be same")
        return ((xs >= self.left) & (xs <= self.right) & (ys <= self.top) & (ys > self.bottom)).nonzero()[0]

    def draw(self, img):
        cv2.rectangle(img, (self.left, self.bottom), (self.right, self.top), self.color, 1)    


class SlideWindow:
    
    def __init__(self, left_lane_l = 50, left_lane_r = 280, right_lane_l = 350, right_lane_r = 560, bottom = 90, top = 300, circle_height = 120):
        self.left_lane_range = WindowRange(top, bottom, left_lane_l, left_lane_r, color = YELLOW)
        self.right_lane_range = WindowRange(top, bottom, right_lane_l, right_lane_r, color = PURPLE)

        self.wd_width = 80
        self.wd_height = 20
        self.stride = 20

        self.circle_height = circle_height
        self.road_width = 0.55

        self.window_minpix = 100
        self.lane_minpix = 300


    def xrange(self, x_val, img):
        y, x = img.shape[0:2]
        return max(0, min(x, x_val))
    
    def yrange(self, y_val, img):
        y, x = img.shape[0:2]
        return max(0, min(y, y_val))
    
    def position_setting_inside_box(self, windowRange, ys, xs, origin_val):

        indexes_in_range = windowRange.indexes_in_range_by_xy(xs, ys)

        if len(indexes_in_range) > self.window_minpix:
            return int(np.mean(xs[indexes_in_range]))
        
        elif len(indexes_in_range) > 2:
            p = np.polyfit(ys[indexes_in_range], xs[indexes_in_range], 2)
            return int(np.polyval(p, windowRange.top))
        
        else:
            return origin_val
        
    def draw_range(self, img, indexes, xs, ys, color):
        for idx in indexes:
            img = cv2.circle(img, (xs[indexes[idx]], ys[indexes[idx]]), 1, color, -1)

    def draw_x_location(self, x_location, img, color):
        cv2.circle(img, (x_location, self.circle_height), 10, color, -1)
        
    def slideWindow(self, img):
        y, x = img.shape[0:2]
        out_img = np.dstack((img, img, img)) * 255
        x_location = x // 2

        self.left_lane_range.draw(img)
        self.right_lane_range.draw(img)

        nonzero = img.nonzero()
        nonzeroy, nonzerox = nonzero[0], nonzero[1]
        left_lane_indexes = self.left_lane_range.indexes_in_range_by_xy(nonzerox, nonzeroy)
        right_lane_indexes = self.right_lane_range.indexes_in_range_by_xy(nonzerox, nonzeroy)


        if len(left_lane_indexes) > self.lane_minpix and len(right_lane_indexes) > self.lane_minpix:
            x_avg_left = int(np.mean(nonzerox[left_lane_indexes]))
            y_top_left = int(np.max(nonzeroy[left_lane_indexes]))

            x_avg_right = int(np.mean(nonzerox[right_lane_indexes]))
            y_top_right = int(np.max(nonzeroy[right_lane_indexes]))

            for top in range(max(y_top_left, y_top_right), 0, -self.stride):

                left_window = WindowRange(top = top,
                                          bottom = self.yrange(top - self.wd_height, img),
                                          left = x_avg_left - (self.wd_width // 2),
                                          right = x_avg_left + (self.wd_width // 2),
                                          color = BLUE )
                
                right_window = WindowRange(top = top,
                                          bottom = self.yrange(top - self.wd_height, img),
                                          left = x_avg_right - (self.wd_width // 2),
                                          right = x_avg_right + (self.wd_width // 2),
                                          color = GREEN )
                
                left_window.draw(img)
                right_window.draw(img)

                x_avg_left = self.position_setting_inside_box(left_window, nonzeroy, nonzerox, x_avg_left)
                x_avg_right = self.position_setting_inside_box(right_window, nonzeroy, nonzerox, x_avg_right)

                if (self.circle_height - self.stride//2 <= left_window.top <= self.circle_height + self.stride//2):
                    x_location = self.xrange((x_avg_left + x_avg_right) // 2, img)

        elif len(left_lane_indexes) > self.lane_minpix and len(right_lane_indexes) < self.lane_minpix:
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
        
        elif len(left_lane_indexes) < self.lane_minpix and len(right_lane_indexes) > self.lane_minpix:
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

        self.draw_x_location(x_location, img, RED)
        return img, x_location
    

if __name__ == "__main__":
    img = cv2.imread("birdeye.png")
    sdw = SlideWindow()
    img, x_location = sdw.slideWindow(img)
    print("x_location: " , x_location)
    cv2.imshow("output", img)
    cv2.waitKey()