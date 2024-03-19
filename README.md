# Hand Segmentation
Model to predict the segmentation of hands in RGB images.

## Prerequisites
- [Python](https://www.python.org/) or [Anaconda](https://www.anaconda.com/)

## Packages

- [PyTorch and torchvision](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Detectron2](https://github.com/facebookresearch/detectron2)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/harryn0502/hand_segmentation.git
cd hand-segmentation
```

### 2. Create a virtual environment

Using conda (recommended)
```bash
conda create -n hand-segmentation python
conda activate hand-segmentation
```

Using venv

```bash
python -m venv venv
```

### 3. Install PyTorch and torchvision
```bash
pip install torch torchvision
```

### 4. Install OpenCV
```bash
pip install opencv-python
```

### 5. Install detectron2
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
**Note**: This may take a while to build and install.

## Inferences / Predictions

### Download the model

Download the pre-trained model from teams and place it in the `model` directory.

### Single Image
```bash
python inference_image.py
```

### Webcam
```bash
python inference_webcam.py
```