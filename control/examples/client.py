from piracer.vehicles import PiRacerStandard, PiRacerPro
from piracer.cameras import Camera, MonochromeCamera

import cv2
import requests
import time

camera = MonochromeCamera()
piracer = PiRacerPro()
url = "http://localhost:5000/process-image"
time_gap = 0.1

if __name__ == "__main__":
    while True:
        start_time = time.time()
        img = camera.read_image()
        _, encoded_image = cv2.imencode('.jpg', img)
        # Convert the encoded image to bytes
        image_bytes = encoded_image.tobytes()
        files = {'image': image_bytes}
        response = requests.post(url, files=files).json()

        end_time = time.time()
        if response.has_key("error"):
            print("ERROR:", response["error"])
        else:
            print("steer:", response["steer"], "speed: ", response["speed"])
            piracer.set_throttle_percent(response["speed"])
            piracer.set_steering_percent(response["steer"])
        time.sleep(time_gap)