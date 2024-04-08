import os
import json
import argparse

images_folder = "images"
mask_folder = "masks"
ground_truth_hand_count_file = "ground_truth/hand_count.json"
estimated_hand_count_file = "output_hand_count/hand_count.json"


#populate hand_count.json (assumed 2 for each)
parser = argparse.ArgumentParser()
parser.add_argument("--default")
args = parser.parse_args()

#test it
if args.default is None:
    with open(ground_truth_hand_count_file, 'r') as f:
        ground_truth_hand_counts = json.load(f)

with open(estimated_hand_count_file, 'r') as f:
    estimated_hand_count = json.load(f)


succesful_tests = 0 
for image, hand_types in sorted(estimated_hand_count.items()):
    if args.default is not None:
        ground_truth = int(args.default)
    else:
        ground_truth = len(ground_truth_hand_counts[image])
    hand_count = len(hand_types)
    if hand_count == ground_truth:
        succesful_tests += 1
        print(f"test {image}: passed")
    elif hand_count < ground_truth:
        print(f"test {image}: failed not enough masks, missing {ground_truth - hand_count} mask(s)")
    else:
        print(f"test {image}: failed too many masks, additional {hand_count - ground_truth} mask(s)")

print(f"total tests passed: {succesful_tests}/{len(estimated_hand_count)}")