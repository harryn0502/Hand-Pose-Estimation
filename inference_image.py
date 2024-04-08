import os
from PIL import Image
from src.predictor import Predictor   
from src.mask_image import black_out_all_bulk, black_out_masks_bulk
from src.estimator.pose_estimator import estimate_pose
import argparse
import json

#clear out a directory, or create it if it does not exist
def clear_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    else:
        for filename in os.listdir(directory):
            for filename in os.listdir(directory):
                os.remove(os.path.join(directory, filename))

# Read arguments
parser = argparse.ArgumentParser()
#optional inversion of mask
parser.add_argument("--invert", action="store_true")
args = parser.parse_args()

# Set the image and output filename
images_folder = "images"
masks_folder = "masks"
masked_images_folder = "masked_images"
output_2d = "output_2d"
output_3d = "output_3d"
output_hand_count = "output_hand_count"
output_hand_count_file = "output_hand_count/hand_count.json"

hand_counts = {}


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
clear_directory(masks_folder)


#dictionnary of images names and their bounding boxes
image_bounding_boxes = {}
for i, image_name in enumerate(image_files):
    image_path = os.path.join(images_folder, image_name)
    image_base = image_name.split('.')[0]
    #initiate empty hand count list
    hand_counts[image_base] = []
    instances = predictor.predict(image_path)
    bounding_boxes = instances.pred_boxes.tensor.numpy()
    
    for j, mask in enumerate(instances.pred_masks):
        mask = mask.to('cpu').numpy()
        image = Image.fromarray(mask)
        image.save(os.path.join(masks_folder, f"{image_base}_mask_{j}.png"))
        bounding_boxes[j][2] -= bounding_boxes[j][0]
        bounding_boxes[j][3] -= bounding_boxes[j][1]
        image_bounding_boxes[f"{image_base}_{j}"] = bounding_boxes[j]

    print(f"segmentation progress {i+1}/{len(image_files)}", end="\r")
print()
print("segmentation completed")

"""
mask images
"""
#delete masks folder and recreate it
clear_directory(masked_images_folder)

#only use 1 masking method (comment other out)

if args.invert:
    #mask all but the mask (no background)
    black_out_all_bulk(images_folder, masks_folder, masked_images_folder)
else:
    #mask all other other masks (with background)
    black_out_masks_bulk(images_folder, masks_folder, masked_images_folder)

print("masking completed")

"""
Pose estimation
"""
clear_directory(output_2d)
clear_directory(output_3d)
clear_directory(output_hand_count)

pose_model = "model/snapshot_20.pth.tar"

files = os.listdir(masked_images_folder)
image_files = sorted([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff'))])

for i, image_name in enumerate(image_files):
    image_base = image_name.split(".")[0]

    image_path = os.path.join(masked_images_folder, image_name)
    output_2d_path = os.path.join(output_2d, image_name)
    output_3d_path = os.path.join(output_3d, image_name)
    hands = estimate_pose(image_path, image_bounding_boxes[image_base], pose_model, output_2d_path, output_3d_path)
    #track hand count and type
    image_group = "_".join(image_base.split("_")[:-1])   
    hand_counts[image_group].extend(hands)

    print(f"pose estimation progress {i+1}/{len(image_files)}", end="\r")

#output hand counts
with open(output_hand_count_file, 'w') as f:
    json.dump(hand_counts, f)

print()
print("pose estimation completed")