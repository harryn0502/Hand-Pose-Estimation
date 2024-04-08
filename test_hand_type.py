import json

ground_truth_hand_count_file = "ground_truth/hand_count.json"
estimated_hand_count_file = "output_hand_count/hand_count.json"

with open(ground_truth_hand_count_file, 'r') as f:
        ground_truth_hand_counts = json.load(f)

with open(estimated_hand_count_file, 'r') as f:
    estimated_hand_count = json.load(f)

success = 0
for image, hand_types in sorted(estimated_hand_count.items()):
        ground_truth_types = ground_truth_hand_counts[image]
        left_hands = sum([1 for hand_type in hand_types if hand_type == "left"])
        left_hands_truth = sum([1 for hand_type in ground_truth_types if hand_type == "left"])
        right_hands = len(hand_types) - left_hands
        right_hands_truth = len(ground_truth_types) - left_hands_truth

        if left_hands == left_hands_truth:
            success += 1
            print(f"{image} left: passed")
        elif left_hands < left_hands_truth:
            print(f"{image} left: failed, missing {left_hands_truth - left_hands} left hands")
        else:
            print(f"{image} left: failed, unexpted additional {left_hands - left_hands_truth} left hand(s)")
             
        if right_hands == right_hands_truth:
            success += 1
            print(f"{image} right: passed")
        elif right_hands < right_hands_truth:
            print(f"{image} right: failed, missing {right_hands_truth - right_hands} right hand(s)")
        else:
            print(f"{image} right: failed, unexpted additional {right_hands - right_hands_truth} right hand(s)")

print(f"Tests passed: {success}/{len(estimated_hand_count)*2}")