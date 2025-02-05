# Team5:ADS - Object Detection and Avoidance

## Project Objective

This project is designed to give you a hands-on experience in developing an object detection and avoidance system for vehicles, using both virtual simulations and real-world hardware implementation. Utilizing platforms like CARLA, Gazebo, or AirSim, you will create a system that not only detects obstacles in a vehicle's path but also performs autonomous maneuvers to avoid potential collisions while ensuring passenger safety.


---

## Project Overview

The files in each directory take the role like:
- **server:** This files have exercisice of providing FLASK server and give calculated output by YOLOPv3 model.
- **colab:** It is a code which is used in Google Colaboratory in Google drive. It read data from Google drive and save the detected lane and segmented road by YOLOPv3 visualization on Google Drive.
- **yolopv3:** They are originally existed files from project YOLOPv3.

---


## Docker Container
Dockerfile for the server is in server/docker_arm64 or server/docker_amd64. If someone don't want to set environment manually, you can just pull and run this Dockerfile. It includes Python dependencies and .pth file of YOLOPv3This. It's how to pull and to run it:

```
docker pull yeongyoo/ads_team:server_flask
docker run --privileged -it --network host -e FLASK_APP=server_yolopv3.py --name ads yeongyoo/ads_team:0.2
```

It's how to use server docker container for server
```
docker start ads
```

## Server Code
It deals with the Flask server code

### server/lib
Basically the python code files inside are same as [this link](https://github.com/jiaoZ7688/YOLOPv3/tree/main/lib), which help to run the model and support various functions.

### server/utils
These code inside aims to extract proper steering and speed value by given code. Deep learning model YOLOPv3 is used for lane segmentation and object detection. The steps are like that:

1. Run the YOLOPv3 model by using class ```ModelRun()``` which includes preprocessing and postprocessing. the output form is such:
```
{
  "identifier": "unique_id",
  "detections": [
    {
      "class": "car",
      "confidence": 0.95,
      "bbox": [xmin, ymin, xmax, ymax],
      "center": [x, y]
    }
  ],
  "da_seg_mask": [[0, 1, 0], [1, 0, 1]],
  "ll_seg_mask": [[1, 0, 1], [0, 1, 0]]
}
```
2. Changing ```ll_seg_mask``` and ```da_seg_mask```into bird eye's view (or vertical view), which makes the movement of , more understandable.
3. Use sliding window way to detect where lanes are and to make robust comprehension how the lane is curved. ```SlidedWindow()``` class returns x-coordinate of center of the road, which can be good clue to notice how curved the lane is.
4. Using value come out from sliding window process, determine the steering and speed value. in this process, if there are objects which is regarded as stop signal or pedestrians, the car must be stopped.

#### image2mani.py
Image2Mani class takes the role of suggesting proper steering and speed value. ```run()``` method in Image2Mani class takes the input as an image and the output is a steering value.

#### modelrun.py
ModelRun class takes the role of running the YOLOPv2 model,
including pre-processing and post-processing, and ```run()``` in ModelRun takes the output as detected objects, segmented driving area, and segmented lane lines area.

#### slidewindow.py
SlideWindow class helps the computer to detect where the lane is, by sliding the fixed-size window. It determines where the lane line is, and give a clue how much the lane is curved.

```slidewindow()``` of slided window takes the input as image and
the output as image with slided window, and x coordinate which is essential clue to know how curved the lane is.

This python file includes, at the same time, WindowRange class, which represents a specific rectangular space in the image, which will be called "Window".

## Contributors
<center>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/jo49973477>">
        <img src="https://github.com/jo49973477.png" width="150px;" alt="Yeongyoo Jo"/>
        <br />
        <sub><b>Yeongyoo Jo</b></sub>
      </a>
      <br />
      <a href="https://github.com/jo49973477"><img src="https://img.shields.io/badge/GitHub-jo49973477-blue?logo=github" alt="GitHub Badge" /></a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/isragogreen">
        <img src="https://github.com/isragogreen.png" width="150px;" alt="Konstantin Tyhomyrov"/>
        <br />
        <sub><b>Konstantin Tyhomyrov</b></sub>
      </a>
      <br />
      <a href="https://github.com/isragogreen"><img src="https://img.shields.io/badge/GitHub-isragogreen-blue?logo=github" alt="GitHub Badge" /></a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/indiks">
        <img src="https://github.com/indiks.png" width="150px;" alt="Sergey Indik"/>
        <br />
        <sub><b>Sergey Indik</b></sub>
      </a>
      <br />
      <a href="https://github.com/indiks"><img src="https://img.shields.io/badge/GitHub-indiks-blue?logo=github" alt="GitHub Badge" /></a>
      <br />
    </td>
  </tr>
</table>
</center>
