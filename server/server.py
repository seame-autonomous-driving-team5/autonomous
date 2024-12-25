# Version: 0

from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO

print("Version: 0")
app = Flask(__name__)
model = YOLO("yolo11n.pt")

# Increase maximum request size to 32 MB
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

@app.route("/process-image", methods=["POST"])
def process_image():
    print("Version 0: Received request to process image.")
    
    # Check if the file is included in the request
    if "image" not in request.files:
        print("Error: No image file provided in the request.")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    try:
        # Read the image from bytes
        file_bytes = np.frombuffer(file.read(), np.uint8)
        print(f"Version 0: Image bytes read, size: {len(file_bytes)}")

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            print("Error: Image decoding failed.")
            return jsonify({"error": "Invalid image file"}), 400

        print(f"Version 0: Image decoded successfully, shape: {image.shape}")

        # Run the model on the image
        results = model(image)
        print(f"Version 0: Model inference completed. Number of results: {len(results)}")

        detections = []
        for result_idx, result in enumerate(results):
            print(f"Version 0: Processing result {result_idx + 1}")
            if result.boxes:
                for box_idx, box in enumerate(result.boxes):
                    cls_idx = int(box.cls.item())
                    class_name = result.names.get(cls_idx, "Unknown")

                    print(f"Version 0: Object {box_idx + 1}: Class: {class_name}, "
                          f"Confidence: {float(box.conf)}, BBox: {box.xyxy.tolist()}")

                    detections.append({
                        "class": class_name,
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()
                    })

        if not detections:
            print("Version 0: No objects detected in the image.")
            return jsonify({"message": "No objects detected", "detections": []}), 200

        print("Version 0: Processing complete. Returning results.")
        return jsonify({"detections": detections}), 200

    except Exception as e:
        print(f"Error: Exception occurred during image processing: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/", methods=["GET"])
def home():
    print("Version 0: GET request received on '/' endpoint.")
    return jsonify({"message": "Server is running!"})

if __name__ == "__main__":
    print("Version 0: Starting Flask server...")
    app.run(host="0.0.0.0", port=5000)
