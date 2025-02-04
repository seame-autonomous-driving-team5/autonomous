import requests
import torch
import numpy as np
import time

url = "http://localhost:5000/process-image"
files = {'image': open('labimage.jpeg', 'rb')}
st = time.time()
response = requests.post(url, files=files).json()

end = time.time()
print(response)
print("time: ", end-st)