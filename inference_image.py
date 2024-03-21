import os
from utils.predictor import Predictor

# Set the image and output filename
file = "sample.jpg"
image = os.path.join("images", file)
filename = os.path.join("output", file)

# Load the model
model = os.path.join("model", "model_final.pth")
predictor = Predictor(model)

# Set the score and IoU threshold (optional)
predictor.set_score_threshold(0.95)
predictor.set_iou_threshold(0.95)

# Predict the image
instances = predictor.predict(image)
predictor.save_image(image, filename, instances)
predictor.save_masks(filename, instances)