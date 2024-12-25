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

## Technical Specification for YOLOPv3 Docker Containers

### Objective

Develop two Docker containers:

1. **For x86 Architecture**: Based on Ubuntu 20.04 LTS.
2. **For Raspberry Pi 4B**: Based on Debian Bookworm.

Each container will:

- Include the YOLOPv3 model with pre-trained weights (`YOLOPv3/epoch-189.pth`), automatically loaded at startup.
- Provide a Flask-based server for image inference, with functionality for:
  - Segmentation masks returned as numerical arrays.
  - JSON responses including detected objects with class names, bounding boxes, and geometric centers.
  - A health-check endpoint to verify the server’s operational status.
- Support Python 3.8 and 3.11 using `pyenv`, allowing runtime selection of Python versions.
- Include all necessary dependencies and fixed versions for reproducibility.
- Be designed for development purposes, with potential for future CUDA support.


### Key Requirements

#### 1. Base Operating Systems

- **x86 Architecture**: Use Ubuntu 20.04 LTS.
- **Raspberry Pi 4B**: Use Debian Bookworm (64-bit).

#### 2. Python and Dependency Management

- Install Python versions 3.8 and 3.11 using `pyenv`.
- Include the following Python dependencies with fixed versions:
  ```
  tensorboardX
  torch==1.12.1
  torchvision
  torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
  yacs
  scipy
  tqdm
  Cython
  matplotlib>=3.2.2
  numpy>=1.18.5
  opencv-python>=4.1.2
  Pillow
  PyYAML>=5.3
  seaborn
  imageio
  scikit-learn
  albumentations
  ptflops
  prefetch-generator
  pretrainedmodels
  h5py
  ```
- Ensure fixed PyTorch versions (no automatic upgrades).
- Add any additional dependencies required by YOLOPv3.

#### 3. Model Integration

- Embed YOLOPv3 weights (`YOLOPv3/epoch-189.pth`) inside the container.
- Implement automatic model weight loading upon container startup.
- Include class mapping for YOLOPv3 to provide meaningful object labels.

#### 4. Flask Server

- Implement the following endpoints:

  1. ``** (POST)**:

     - Accept an image file (640x640x3) and a unique identifier.
     - Perform inference using YOLOPv3:
       - Return detected objects with:
         - Class name
         - Confidence score
         - Bounding box coordinates (xmin, ymin, xmax, ymax)
         - Geometric center (x, y)
       - Return segmentation masks (`da_seg_out` and `ll_seg_out`) as numerical arrays.
     - Response JSON format:
       ```json
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

  2. ``** (GET)**:

     - Return server status as:
       ```json
       {
         "status": "Server is running"
       }
       ```

- Configure Flask in development mode for testing.

#### 5. CUDA Support

- CUDA is not required initially but should be planned for future compatibility.
- Ensure the container setup can be extended for CUDA (e.g., in Google Colab).

#### 6. Dockerfile Specifications

- Create two separate Dockerfiles:

  1. **For x86 Architecture**:
     - Base image: Ubuntu 20.04 LTS.
     - CPU-only PyTorch version from the official index.
  2. **For Raspberry Pi 4B**:
     - Base image: Debian Bookworm (64-bit).
     - CPU-only PyTorch version compatible with ARM64.

- Install all dependencies and system packages (e.g., `pyenv`, OpenCV, Flask).

- Configure the environment for Python runtime selection via `pyenv`.

- Embed model weights and ensure automatic loading at startup.

- Optimize image size (use multi-stage builds if necessary).

#### 7. Testing and Validation

- Verify the Flask server endpoints for functionality:
  - `/process-image`: Test with valid and invalid images.
  - `/health`: Confirm proper server status response.
- Validate model inference accuracy and consistency across both architectures.
- Test segmentation mask outputs as numerical arrays.


### Deliverables

1. Two Dockerfiles:
   - `Dockerfile.x86`: For Ubuntu 20.04 LTS (x86).
   - `Dockerfile.arm64`: For Debian Bookworm (Raspberry Pi 4B).
2. Flask server implementation.
3. Documentation:
   - Instructions for building and running the containers.
   - API documentation for the Flask server.
4. Sample test scripts for validating endpoints and model outputs.

### Future Considerations

- Add CUDA support for Google Colab or other environments.
- Optimize Docker images for deployment (e.g., smaller base images, multi-stage builds).

---

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

