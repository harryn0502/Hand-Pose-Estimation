import os
from utils.predictor import Predictor

file = "sample.jpg"
image = os.path.join("images", file)
filename = os.path.join("output", file)

predictor = Predictor()

# Set the score and IoU threshold (optional)
# predictor.set_score_threshold(0.5)
# predictor.set_iou_threshold(0.95)

instances = predictor.predict(image)
predictor.save_image(image, filename, instances)
predictor.save_masks(filename, instances)