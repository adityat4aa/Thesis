import os
import glob
import cv2
import pybboxes as pbx
import yaml
import argparse
import shutil
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
import dill._dill as dill

# Safe deserialization setup
torch.serialization.add_safe_globals([DetectionModel, dill._load_type])

# Parse config path
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to configuration YAML file", required=True)
args = parser.parse_args()

# Load YAML config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Clean old annotation folder if exists
if os.path.exists("annot_txt"):
    shutil.rmtree("annot_txt")
os.makedirs("annot_txt")

# Load YOLO model
with torch.serialization.safe_globals([DetectionModel]):
    model = YOLO(config["model_path"])

# Run detection
_ = model(
    source=config["images_path"],
    save=False,
    save_txt=True,
    conf=config["detection_conf_thresh"],
    device='cuda:0' if config["gpu_avail"] else 'cpu',
    project="runs/detect/",
    name="yolo_images_pred"
)

# Convert YOLO bbox format to VOC and save in annot_txt/
annot_dir = "runs/detect/yolo_images_pred/labels/"
image_size = (config["img_width"], config["img_height"])

for file in os.listdir(annot_dir):
    if file.endswith('.txt'):
        with open(os.path.join(annot_dir, file), 'r') as fin:
            lines = fin.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            bbox_yolo = [float(i) for i in parts[1:]]
            bbox_voc = pbx.convert_bbox(bbox_yolo, from_type="yolo", to_type="voc", image_size=image_size)
            with open(f"annot_txt/{file}", "a") as fout:
                fout.write(" ".join(str(int(coord)) for coord in bbox_voc) + "\n")

# Blurring function
def blur_regions(image, regions):
    if image is None:
        return image
    for region in regions:
        x1, y1, x2, y2 = map(int, region)
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        blurred_roi = cv2.GaussianBlur(roi, (config["blur_radius"], config["blur_radius"]), 0)
        image[y1:y2, x1:x2] = blurred_roi
    return image

# Process and blur each image
txt_folder = "annot_txt/"
image_folder = config["images_path"]
output_folder = config["output_folder"]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

for txt_file in txt_files:
    bboxes = []
    with open(os.path.join(txt_folder, txt_file), 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        x_min, y_min, x_max, y_max = map(int, parts)
        bboxes.append([x_min, y_min, x_max, y_max])

    if not bboxes:
        print(f"⚠️ No bounding boxes found in {txt_file}. Skipping this image.")
        continue

    image_file = txt_file.replace('.txt', config["img_format"])
    image_path = os.path.join(image_folder, image_file)

    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        continue

    blurred_image = blur_regions(image, bboxes)

    output_file = txt_file.replace('.txt', '_blurred.jpg')
    output_path = os.path.join(output_folder, output_file)
    cv2.imwrite(output_path, blurred_image)

print(f"✅ Blurred images saved to: {config['output_folder']}")

