import os
import cv2
import torch
import numpy as np
import traceback

from flask import Flask, request, jsonify
from torchvision import transforms
from lib.config import cfg
from lib.models import get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils.utils import select_device
from lib.core.postprocess import morphological_process, connect_lane
from lib.utils import show_seg_result

# Flask application setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

# model input picture size
model_size = (640, 640)

# Load YOLOPv3 model
device = select_device(None, "0")  # Use GPU 0 by default

model = get_net(cfg)
checkpoint = torch.load("epoch-189.pth", map_location=device)  # Replace with weights path
model.load_state_dict(checkpoint['state_dict'])
model.to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(model_size),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        
        # Prepare image for model
        print(image.shape)
        img_tensor = transform(image).unsqueeze(0).to(device)
        print(img_tensor.size())

        # Run inference
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(img_tensor)

        # Post-process detection results
        det_pred = non_max_suppression(det_out[0], conf_thres=0.3, iou_thres=0.45)[0]
        detections = []
        if det_pred is not None:
            det_pred[:, :4] = scale_coords(img_tensor.shape[2:], det_pred[:, :4], image.shape).round()
            for *xyxy, conf, cls in det_pred:
                detections.append({
                    "bbox": [int(coord) for coord in xyxy],
                    "confidence": float(conf),
                    "class": int(cls)
                })

        # Process segmentation masks
        da_seg_mask = torch.nn.functional.interpolate(
            da_seg_out[:, :, :, :], size=(720, 1280), mode='bilinear'
        ).argmax(dim=1).squeeze().cpu().numpy()

        ll_seg_mask = torch.nn.functional.interpolate(
            ll_seg_out[:, :, :, :], size=(720, 1280), mode='bilinear'
        ).argmax(dim=1).squeeze().cpu().numpy()

        # Prepare response
        response = {
            "detections": detections,
            "segmentation": {
                "drivable_area": da_seg_mask.tolist(),
                "lane_lines": ll_seg_mask.tolist()
            }
        }
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