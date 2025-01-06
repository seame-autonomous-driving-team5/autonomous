import requests
import torch
import numpy as np

url = "http://localhost:5000/process-image"
files = {'image': open('blackbox.jpg', 'rb')}
response = requests.post(url, files=files).json()

print("detections: ", response["detections"])
print("drivable area: ", np.array(response["segmentation"]["drivable_area"]))
print("lane_lines: ",  np.array(response["segmentation"]["lane_lines"]))