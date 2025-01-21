import os
import cv2
import torch
import numpy as np
import traceback

from flask import Flask, request, jsonify

from utils.modelrun import ModelRun
from utils.image2mani import Image2Mani
 
# Flask application setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

model_run = ModelRun()
img2main = Image2Mani(mode = "extreme", speed = 0.1)

@app.route("/process-image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        # Decode image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        
        #run it and see the result
        steer, speed = img2main.run(image)
        response = {"steer": steer, "speed": speed}
        return jsonify(response), 200

    except Exception as e:
        error_traceback = traceback.format_exc()
        
        return jsonify({"error": str(e),
                        "traceback": error_traceback}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "YOLOPv3 Flask server is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)