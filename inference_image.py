import os
from src.predictor import Predictor   
from src.mask_image import black_out_region_bulk
from src.pose_estimator.estimate_pose import estimate_pose

# Set the image and output filename
images_folder = "images"
masks_folder = "masks"
masked_images_folder = "masked_images"
output_folder = "output"


"""
Segmentation
"""
# Load the model
segmentation_model = "model/model_final.pth"
predictor = Predictor(segmentation_model)

# Set the score and IoU threshold (optional)
predictor.set_score_threshold(0.95)
predictor.set_iou_threshold(0.95)

# Predict the images
files = os.listdir(images_folder)
image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff'))]

if not os.path.isdir(masks_folder):
    os.makedirs(masks_folder)

for image_name in image_files:
    image_path = os.path.join(images_folder, image_name)
    output_name = os.path.join(masks_folder, image_name)

    instances = predictor.predict(image_path)
    #predictor.save_image(image_path, output_name, instances)
    predictor.save_masks(output_name, instances)

"""
mask images
"""
#delete masks folder and recreate it
if not os.path.isdir(masked_images_folder):
    os.makedirs(masked_images_folder)

black_out_region_bulk(images_folder, masks_folder, masked_images_folder)

"""
Pose estimation
"""
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

pose_model = "model/snapshot_99.pth.tar"
estimate_pose(masked_images_folder, output_folder, pose_model)