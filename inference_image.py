import os
from utils.predictor import Predictor

file = "imagehand8.jpg"
image = os.path.join("images", "custom", file)
filename = os.path.join("output", file)

predictor = Predictor()

# Set the score and IoU threshold (optional)
predictor.set_score_threshold(0.97)
predictor.set_iou_threshold(0.95)

instances = predictor.predict(image)
predictor.save_image(image, filename, instances)
predictor.save_masks(filename, instances)