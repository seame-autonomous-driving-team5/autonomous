import os
import cv2
import torch
import numpy as np
import traceback
 
import sys
sys.path.append('..')

from flask import Flask, request, jsonify
from torchvision import transforms
from lib.config import cfg
from lib.models import get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils.utils import select_device
from lib.core.postprocess import morphological_process, connect_lane
from lib.utils import show_seg_result

class ModelRun:
    def __init__(self, model_size = (640, 640)):
        # model input picture size
        self.model_size = model_size

        # Load YOLOPv3 model
        self.device = select_device(None, "0" if torch.cuda.is_available() else "cpu")  # Use GPU 0 by default

        self.model = get_net(cfg)
        checkpoint = torch.load("epoch-189.pth", map_location=self.device, weights_only=True)  # Replace with weights path
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.model_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def run(self, img):
        y, x = img.shape[0:2]

        # pre-processing before running the model
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # run the model
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = self.model(img_tensor)
        
        # conducting NMS after running
        det_pred = non_max_suppression(det_out[0], conf_thres=0.3, iou_thres=0.45)[0]
        detections = []
        
        # saving the data as a form of dictionary
        if det_pred is not None:
            det_pred[:, :4] = scale_coords(img_tensor.shape[2:], det_pred[:, :4], img.shape).round()
            for *xyxy, conf, cls in det_pred:
                detections.append({
                    "bbox": [int(coord) for coord in xyxy],
                    "confidence": float(conf),
                    "class": int(cls)
                })

        # let the segmented data return to the normal
        da_seg_mask = torch.nn.functional.interpolate(
            da_seg_out[:, :, :, :], size=(y, x), mode='bilinear'
        ).argmax(dim=1).squeeze().cpu().numpy()

        ll_seg_mask = torch.nn.functional.interpolate(
            ll_seg_out[:, :, :, :], size=(y, x), mode='bilinear'
        ).argmax(dim=1).squeeze().cpu().numpy()

        # Prepare response
        response = {
            "detections": detections,
            "segmentation": {
                "drivable_area": da_seg_mask.tolist(),
                "lane_lines": ll_seg_mask.tolist()
            }
        }

        return response
