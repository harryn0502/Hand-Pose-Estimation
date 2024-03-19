# SUPPRESS WARNINGS
import warnings
warnings.filterwarnings("ignore") 

# COMMON LIBRARIES
import os
import cv2
import torch
import time
from itertools import combinations

# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Instances, Boxes
from PIL import Image
import matplotlib.pyplot as plt

# CONFIGURATION
from detectron2 import model_zoo
from detectron2.config import get_cfg

# INFERENCE
from detectron2.engine import DefaultPredictor

# HYPERPARAMETERS
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
NUM_CLASSES = 1
SCORE_THRESHOLD = 0.5
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# CONFIGURATION
CFG = get_cfg()
CFG.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
CFG.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
CFG.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
CFG.MODEL.WEIGHTS = os.path.join("model", "model_final.pth")
CFG.MODEL.DEVICE = DEVICE

# DATASET METADATA
class Metadata:
    def get(self, _):
        return ['hand'] #your class labels

class Predictor:
    def __init__(self, score_threshold=0.5, iou_threshold=0.95):
        self.cfg = CFG.clone()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        self.predictor = DefaultPredictor(self.cfg)
        self.iou_threshold = iou_threshold
        self.instances = None

    def set_score_threshold(self, score_threshold):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        self.predictor = DefaultPredictor(self.cfg)

    def set_iou_threshold(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def predict(self, image_path):
        img = cv2.imread(image_path)
        outputs = self.predictor(img)
        self.instances = self._remove_overlapping_masks(outputs["instances"])
        return self.instances
    
    def save_image(self, image, filename, instances):
        output_path = filename.split("/")[0]
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        img = cv2.imread(image)
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=Metadata, 
            scale=0.8, 
            instance_mode=ColorMode.IMAGE_BW
        )

        out = visualizer.draw_instance_predictions(instances.to("cpu"))
        plt.imsave(filename, out.get_image()[:, :, ::-1])

    def save_masks(self, filename, instances):
        path = filename.split("/")
        output_path = path[0]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for id, mask in enumerate(instances.pred_masks):
            mask = mask.to('cpu').numpy()
            image = Image.fromarray(mask)
            name = path[1].split(".")[0]
            image.save(os.path.join(output_path, f"{name}_mask_{id}.png"))

    def start_webcam(self):
        # 0 is the default camera, change it if you have multiple cameras
        cap = cv2.VideoCapture(1)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            results = self.predictor(frame)
            end = time.perf_counter()
            total_time = end - start
            fps = 1 / total_time

            visualizer = Visualizer(
                frame[:, :, ::-1],
                metadata=Metadata, 
                scale=0.8, 
                instance_mode=ColorMode.IMAGE_BW
            )

            instances = self._remove_overlapping_masks(results["instances"])
            out = visualizer.draw_instance_predictions(instances.to("cpu"))
            image = out.get_image()[:, :, ::-1].astype("uint8")
            cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Webcam", image)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def _remove_overlapping_masks(self, instances):
        # Compare all the masks with each other and filter the overlapping ones
        filter_list = set()
        for i, j in combinations(range(len(instances.pred_masks)), 2):
            mask1 = instances.pred_masks[i]
            mask2 = instances.pred_masks[j]
            iou = self._get_iou(mask1, mask2)
            
            # If the IOU is greater than the threshold, filter the mask with smaller area
            if iou > self.iou_threshold:
                if mask1.sum() < mask2.sum():
                    filter_list.add(i)
                else:
                    filter_list.add(j)

        filtered_list = set(range(len(instances))) - filter_list

        if len(filtered_list) == 0:
            return instances
    
        # Create a new instance with the filtered masks
        new_instances = Instances(instances.image_size)
        new_instances.set("pred_boxes", Boxes(torch.stack([instances.pred_boxes.tensor[i] for i in filtered_list])))
        new_instances.set("scores", torch.stack([instances.scores[i] for i in filtered_list]))
        new_instances.set("pred_classes", torch.stack([instances.pred_classes[i] for i in filtered_list]))
        new_instances.set("pred_masks", torch.stack([instances.pred_masks[i] for i in filtered_list]))
        return new_instances
    
    def _get_iou(self, mask1, mask2):
        intersection = (mask1 * mask2).sum()
        union = (mask1 + mask2).sum()
        return intersection / union