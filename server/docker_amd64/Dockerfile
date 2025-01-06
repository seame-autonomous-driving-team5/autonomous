FROM python:3.12

# git clone the python file I need.
WORKDIR /root
RUN apt update && apt install -y git libgl1
RUN pip install gdown opencv-python
RUN git clone https://github.com/seame-autonomous-driving-team5/autonomous

# install dependencies related to PyTorch
WORKDIR /root/autonomous/server
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN gdown --fuzzy https://drive.google.com/file/d/1JYMI9VXKYoGgOJD22_bAfUT6XdiX8Mgj/view?usp=sharing

# download the pth file


CMD ["python3", "server_yolopv3.py"]
