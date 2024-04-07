import os
import json
import argparse

images_folder = "images"
mask_folder = "masks"
hand_count_file = "ground_truth/hand_count.json"

files = os.listdir(images_folder)
image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.tiff'))]




#populate hand_count.json (assumed 2 for each)
parser = argparse.ArgumentParser()
parser.add_argument("--default")
args = parser.parse_args()

if args.default is not None:
    default_hands = args.default
    hand_counts = {}
    for image_file in image_files:
        hand_counts[image_file.split(".")[0]] = int(default_hands)
    with open(hand_count_file, 'w') as f:
        json.dump(hand_counts, f)


#test it
with open(hand_count_file, 'r') as f:
    hand_counts = json.load(f)


mask_files = os.listdir(mask_folder)

for mask_file in mask_files:
    hand_counts[mask_file.split("_mask")[0]] -= 1

succesful_tests = 0 
for image, hand_count in sorted(hand_counts.items()):
    if hand_count == 0:
        succesful_tests += 1
        print(f"test {image}: passed")
    elif hand_count > 0:
        print(f"test {image}: failed not enough masks, missing {hand_count} mask(s)")
    else:
        print(f"test {image}: failed too many masks, additional {-hand_count} mask(s)")

print(f"total tests passed: {succesful_tests}/{len(hand_counts)}")