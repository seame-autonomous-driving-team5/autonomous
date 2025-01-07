# Base image with PyTorch pre-installed
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /root

# Install additional dependencies
RUN apt update && apt install -y \
    git \
    libgl1 && \
    pip install gdown opencv-python

# Clone the repository
RUN git clone https://github.com/seame-autonomous-driving-team5/autonomous

# Install Python dependencies
WORKDIR /root/autonomous/server
RUN pip install -r requirements.txt

# Download pre-trained model file (if required)
RUN gdown --fuzzy https://drive.google.com/file/d/1JYMI9VXKYoGgOJD22_bAfUT6XdiX8Mgj/view?usp=sharing

# Command to run the server
CMD ["python3", "server_yolopv3.py"]
