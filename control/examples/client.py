from piracer.vehicles import PiRacerStandard, PiRacerPro

import cv2
import requests
import time
from picamera2 import Picamera2
import io
from PIL import Image


piracer = PiRacerStandard()
picam2 = Picamera2()
url = "http://localhost:5000/process-image"
time_gap = 0.1

if __name__ == "__main__":
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    while True:
        start_time = time.time()
        img = picam2.capture_array()
        _, encoded_image = cv2.imencode('.jpg', img)
        # Convert the encoded image to bytes
        image_data = io.BytesIO(encoded_image)
        files = {'image': image_data}
        response = requests.post(url, files=files).json()

        end_time = time.time()
        if response.has_key("error"):
            print("ERROR:", response["error"])
        else:
            print("steer:", response["steer"], "speed: ", response["speed"])
            piracer.set_throttle_percent(response["speed"])
            piracer.set_steering_percent(response["steer"])
        time.sleep(time_gap)