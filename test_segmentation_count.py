import os
import json

images_folder = "images"
mask_folder = "masks"
hand_count_file = "ground_truth/hand_count.json"

files = os.listdir(images_folder)
image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff'))]

#populate hand_count.json (assumed 2 for each)
hand_counts = {}
for image_file in image_files:
    hand_counts[image_file.split(".")[0]] = 2
with open(hand_count_file, 'w') as f:
    json.dump(hand_counts, f)


#test it
with open(hand_count_file, 'r') as f:
    hand_counts = json.load(f)


mask_files = os.listdir(mask_folder)

for mask_file in mask_files:
    hand_counts[mask_file.split("_mask")[0]] -= 1

for image, hand_count in sorted(hand_counts.items()):
    if hand_count == 0:
        print(f"test {image}: passed")
    elif hand_count > 0:
        print(f"test {image}: failed not enough masks, missing {hand_count} mask(s)")
    else:
        print(f"test {image}: failed too many masks, additional {-hand_count} mask(s)")
