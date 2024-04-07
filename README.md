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
conda create -n hand-segmentation python=3.9
conda activate hand-segmentation
```


### 3. Install Dependencies

```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

```bash
pip install -r requirements.txt
```

**Note**: This may take a while to build and install.

## Inferences / Predictions

### Download the model

Download the pre-trained segementation model "model_final.pth" and pose estimation `snapshot_99.pth.tar` and put it into the model directory.

### Single Image
```bash
python inference_image.py
```

### Webcam
```bash
python inference_webcam.py
```