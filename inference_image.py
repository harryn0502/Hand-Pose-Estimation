import os
from src.predictor import Predictor   
from src.mask_image import black_out_region_bulk
from src.pose_estimator.estimate_pose import estimate_pose

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

clear_directory(masks_folder)


for i, image_name in enumerate(image_files):
    image_path = os.path.join(images_folder, image_name)
    output_name = os.path.join(masks_folder, image_name)

    if image_name != "rgb_0002.png":
        continue

    instances = predictor.predict(image_path)

    if image_name == "rgb_0002.png":
        for i in range(len(instances.pred_boxes)):
            print(instances.pred_boxes[i])
            # print(instances.pred_boxes[i].numpy())
        breakpoint()
    #predictor.save_image(image_path, output_name, instances)
    predictor.save_masks(output_name, instances)
    print(f"segmentation progress {i+1}/{len(image_files)}", end="\r")
print()
print("segmentation completed")

"""
mask images
"""
#delete masks folder and recreate it
clear_directory(masked_images_folder)

black_out_region_bulk(images_folder, masks_folder, masked_images_folder)

print("masking completed")

"""
Pose estimation
"""
clear_directory(output_folder)

pose_model = "model/snapshot_99.pth.tar"
estimate_pose(masked_images_folder, output_folder, pose_model)

print("pose estimation completed")