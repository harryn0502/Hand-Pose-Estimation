import os
from PIL import Image
from src.predictor import Predictor   
from src.mask_image import black_out_all_bulk, black_out_masks_bulk
from src.estimator.pose_estimator import estimate_pose
import argparse

#clear out a directory, or create it if it does not exist
def clear_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    else:
        for filename in os.listdir(directory):
            for filename in os.listdir(directory):
                os.remove(os.path.join(directory, filename))

# Set the image and output filename
images_folder = "images"
output_2d = "output_2d/without_segmentation"
output_3d = "output_3d/without_segmentation"

# Predict the images
files = os.listdir(images_folder)
image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff'))]

#dictionnary of images names and their bounding boxes
image_bounding_boxes = {}
for i, image_name in enumerate(image_files):
    image_path = os.path.join(images_folder, image_name)
    image_base = image_name.split('.')[0]
    img = Image.open(image_path)
    width, height = img.size
    image_bounding_boxes[image_base] = [0, 0, width, height]

"""
Pose estimation
"""
clear_directory(output_2d)
clear_directory(output_3d)

pose_model = "model/snapshot_20.pth.tar"

for i, image_name in enumerate(image_files):
    image_base = image_name.split(".")[0]
    image_path = os.path.join(images_folder, image_name)
    output_2d_path = os.path.join(output_2d, image_name)
    output_3d_path = os.path.join(output_3d, image_name)
    estimate_pose(image_path, image_bounding_boxes[image_base], pose_model, output_2d_path, output_3d_path, mode="double")

    print(f"pose estimation progress {i+1}/{len(image_files)}", end="\r")
print()
print("Pose Estimation Completed")