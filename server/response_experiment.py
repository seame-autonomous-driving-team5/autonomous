import requests
import torch
import numpy as np

url = "http://localhost:5000/process-image"
files = {'image': open('labimage.jpeg', 'rb')}
response = requests.post(url, files=files).json()
 
print(response)