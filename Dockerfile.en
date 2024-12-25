# Use the base image from Ultralytics
FROM ultralytics/ultralytics:latest-arm64

# Install additional dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Update pip (if needed)
RUN pip install --upgrade pip setuptools wheel

# Copy the dependencies file into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies from requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the application into the container
COPY . /app
WORKDIR /app

# Specify the command to start the Flask server
CMD ["python", "server.py"]
