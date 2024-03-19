
import os
from utils.predictor import Predictor

file = "sample.jpg"
image = os.path.join("images", file)
filename = os.path.join("output", file)

predictor = Predictor()
instances = predictor.predict(image)
predictor.save_image(image, filename, instances)
predictor.save_masks(filename, instances)