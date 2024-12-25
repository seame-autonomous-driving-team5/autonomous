# https://colab.research.google.com/drive/1DvjVgJW5StD9M-WFBt4oKEZFk-Raeykm?usp=sharing
import os
import sys
import json
import cv2
import torch
import shutil
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
from numpy import random

# Paths on Google Drive
input_folder =  "/content/drive/MyDrive/VID/DEMO_IN"
output_folder = "/content/drive/MyDrive/VID/DEMO_OUT"
progress_file = "/content/drive/MyDrive/VID/progress.json"
project_path =  "/content/YOLOPv3"
weights_file =  "/content/YOLOPv3/epoch-189.pth"

# Image processing settings
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])
img_size = 640

# Checking required paths
def check_required_paths():
    paths = [input_folder, project_path, weights_file]
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path is missing: {path}")
    os.makedirs(output_folder, exist_ok=True)

# Reading progress from JSON
def read_progress():
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(json.load(f).get('processed_files', []))
    return set()

# Updating progress
def update_progress(processed_files):
    with open(progress_file, 'w') as f:
        json.dump({"processed_files": list(processed_files)}, f)

# Getting list of images including subfolders
def get_images_in_folder(folder_path, extensions=('.jpg', '.jpeg', '.png'), max_depth=5):
    images = []
    base_path = Path(folder_path)

    for root, _, files in os.walk(folder_path):
        # Check depth of nesting
        depth = Path(root).relative_to(base_path).parts
        if len(depth) > max_depth:
            continue

        # Filter files by extensions
        for file in files:
            if file.lower().endswith(extensions):
                images.append((root, file))

    return images

# Loading the model
def load_model(cfg, weights, device):
    sys.path.append(project_path)
    from lib.config import cfg as lib_cfg, update_config
    from lib.models import get_net
    from lib.core.general import non_max_suppression, scale_coords
    from lib.utils import plot_one_box, show_seg_result

    model = get_net(cfg)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
#    model.fuse()  # Optional: Fuse layers for inference optimization
    model = model.to(device)
    model.eval()
    return model, non_max_suppression, scale_coords, plot_one_box, show_seg_result

# Main function for image processing
def detect(cfg, opt, processed_files):
    logger = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model, non_max_suppression, scale_coords, plot_one_box, show_seg_result = load_model(cfg, opt.weights, device)

    # Check and create output directory
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)  # Create folder if it doesn't exist

    # Get list of all images
    images_list = get_images_in_folder(opt.source, max_depth=5)

    # Class names and colors for visualization
    names = model.names if hasattr(model, 'names') else model.module.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Inference
    model.eval()
    processed_count = 0  # Counter for processed files
    for root, file_name in tqdm(images_list, total=len(images_list)):
        input_path = os.path.join(root, file_name)

        # Skip already processed files
        if input_path in processed_files:
            continue

        # Load the image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to load image: {input_path}")
            continue

        img_det = img.copy()
        img = cv2.resize(img, (opt.img_size, opt.img_size))
        img = transform(img).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(img)
            inf_out, _ = det_out
            det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)

        # Segmentation masks
        h, w, _ = img_det.shape
        da_seg_mask = torch.nn.functional.interpolate(
            da_seg_out, size=(h, w), mode='bilinear').argmax(1).cpu().numpy()[0]

        ll_seg_mask = torch.nn.functional.interpolate(
            ll_seg_out, size=(h, w), mode='bilinear').argmax(1).cpu().numpy()[0]

        # Visualize results
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        if len(det_pred[0]):
            det_pred[0][:, :4] = scale_coords(img.shape[2:], det_pred[0][:, :4], img_det.shape).round()
            for *xyxy, conf, cls in reversed(det_pred[0]):
                label = f"{names[int(cls)]} {conf:.2f}"
                plot_one_box(xyxy, img_det, label=label, color=colors[int(cls)], line_thickness=2)

        # Save result while maintaining folder structure
        relative_path = Path(root).relative_to(opt.source)
        output_dir = os.path.join(opt.save_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"{Path(file_name).stem}_result.jpg")
        cv2.imwrite(save_path, img_det)
        print(f"folder/file save- ", save_path)

        # Add file to processed
        processed_files.add(input_path)
        processed_count += 1

        # Save progress every 10 files
        if processed_count % 10 == 0:
            update_progress(processed_files)

    # Final save of progress
    update_progress(processed_files)

# Main program block
if __name__ == "__main__":
    check_required_paths()

    # Read progress
    processed_files = read_progress()

    # Settings
    class Opt:
        source = input_folder
        save_dir = "/content/drive/MyDrive/VID/DEMO_OUT"    # output_folder
        weights = weights_file
        img_size = img_size
        conf_thres = 0.35 #0.30
        iou_thres = 0.40  #0.45

    opt = Opt()

    # Perform inference
    detect(cfg=None, opt=opt, processed_files=processed_files)

    # Update progress
    update_progress(processed_files)

    print("Processing completed.")
