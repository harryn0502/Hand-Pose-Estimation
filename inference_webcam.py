from utils.predictor import Predictor

predictor = Predictor()

# Set the score and IoU threshold (optional)
predictor.set_score_threshold(0.97)
predictor.set_iou_threshold(0.95)

predictor.start_webcam()