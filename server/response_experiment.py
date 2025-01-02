import requests
import torch
import numpy

url = "http://localhost:5000/process-image"
files = {'image': open('blackbox.jpg', 'rb')}
response = requests.post(url, files=files).json()

print("detections: ", response["detections"])