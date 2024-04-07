# Hand Segmentation
Model estimate the 3d pose of multiple hands from an rgb image.

## Prerequisites
- [Python](https://www.python.org/) or [Anaconda](https://www.anaconda.com/)

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

Download the pre-trained segementation model `model_final.pth`` and pose estimation `snapshot_20.pth.tar` and put them into the model directory.

### Estimate from images folder
```bash
python inference_image.py
```

### Test segmentation
To test whether the correct amount of hands were detected for each image after running inference_image.py

Add a hand_count.json file into the ground_truth folder then run:
```bash
python test_segmentation_count.py
```

Optionally if all images have the same amount of hands you can add the `--default` flag and specify an integer afterwards, for example for an expected amount of 2 hands in each image we would run:

```bash
python test_segmentation_count.py --default 2
```
