# Copyright (C) 2022 twyleg
import cv2
import requests
from piracer.cameras import Camera, MonochromeCamera

if __name__ == '__main__':
    camera = MonochromeCamera()
    # camera = Camera()

    image = camera.read_image()
    cv2.imwrite('image.png', image)

