# 1. Installation
Create environment
```
conda create -n double_hand python=3.9
conda activate double_hand 
```
Install pytorch
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Install other packages
```
pip install -r requirements.txt
```
# 2. Demo
## Download models
Download pre-trained model "snapshot_99.pth.tar" from Teams and put it into folder `demo/`.
The model is trained on InterHand2.6M(v1.0) including images of 5fps.

The `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` are already in the `mano/` directory

Go to `demo/` and run the code
```
python demo.py --test_epoch 99
```
The model will predict hand shapes from images and estimate meshes in `test_folder/`.
