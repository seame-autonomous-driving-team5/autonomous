import os
import sys
os.environ["QT_QPA_PLATFORM"] = "xcb"

from math import atan2, tanh
import cv2
import numpy as np

sys.path.append('..')

from modelrun import ModelRun
from slidewindow import SlideWindow

class Image2Mani():
    def __init__(self, mode = "extreme", speed = 0.1):
        print("model")
        self.modelrun = ModelRun()
        self.slidewindow = SlideWindow()

        # for changing bird eye's view
        self.src_horizon = 330
        self.src_margin_down = 50
        self.src_margin_xaxis = 120

        self.dst_margin_up = 50
        self.dst_margin_down = 50

        # speed setting
        self.speed = speed

        # set exterme steering value or just using normal value
        if mode not in ["extreme", "normal", "tanh"]:
            raise ValueError("You must set mode among 'extreme', 'normal', 'tanh'. ")
        self.mode = mode
        self.threshold = 50

    def binary2img(self, bin):
        img = cv2.normalize(bin, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img = img.astype(np.uint8)
        return img
    
    def img2binary(self, img):
        bin = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        return bin
        
    def bird_eyes_view(self, img):

        img = self.binary2img(img)

        print(img.shape)
        y, x = img.shape[0:2]


        # values for changing bird_eye's view
        src = np.float32([ # left-top , left-down, right-top, right-down
            [self.src_margin_xaxis, self.src_horizon],
            [0, y - self.src_margin_down],
            [x - self.src_margin_xaxis, self.src_horizon],
            [x, y - self.src_margin_down]
        ])

        dst =  np.float32([ # left-top , left-down, right-top, right-down
            [self.dst_margin_up, 0],          
            [self.dst_margin_down, y],           
            [x - self.dst_margin_up, 0],  
            [x - self.dst_margin_down, y] 
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(img, matrix, (x,y))
        return warped_img
    
    def determine_steer(self, x_location, img):
        y, x = img.shape[0:2]
        cte = x_location - x//2

        if self.mode == "extreme":
            if - self.threshold <= cte <= self.threshold:
                steer = 0
            else:
                steer = 1 if cte >= self.threshold else -1

        elif self.mode == "tanh":
            steer = tanh(cte / self.threshold)

        else:
            steer =  atan2(cte , y - self.threshold) / (np.pi / 2)
            

        return steer

        
    def run(self, img):

        response = self.modelrun.run(img)

        da_birdeye = self.bird_eyes_view(np.array(response["segmentation"]["drivable_area"]))
        ll_birdeye = self.bird_eyes_view(np.array(response["segmentation"]["lane_lines"]))
        cv2.imwrite("birdeye.png", ll_birdeye)

        slided_img, x_location, = self.slidewindow.slideWindow(ll_birdeye)
        cv2.imwrite("slided.png", slided_img)

        steer = self.determine_steer(x_location, img)

        return steer, self.speed
    


if __name__ == "__main__":
    img = cv2.imread("../labimage.jpeg")
    modelrun = Image2Mani(mode = "tanh")
    steer, speed = modelrun.run(img)
    print("steer:", steer, "speed:", speed)
    cv2.waitKey()