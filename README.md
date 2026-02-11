## CNN-based Material Detection System

This project implements a CNN-based material classification and real-time detection system. It can be trained on your custom dataset of materials (e.g., Glass, Metal, Paper, Plastic) and then used on a regular PC or a Raspberry Pi for live detection.

### Setup

- **Install dependencies**:

```bash
pip install -r requirements.txt
```

- **Dataset location and structure**  
Provide **your own dataset path** (a folder that contains one subfolder per class, e.g. `Glass`, `Metal`, `Paper`, `Plastic`) and make sure the code points to it.

You can add more classes by creating additional folders inside `Dataset`.  
If your dataset is in a different location, update:

- `BASE_DATASET_DIR` in `train_and_detect.py`, and/or
- `dataset_paths` in `train.py`.

### Usage

- **Option 1 – Simple training + detection in one script (basic CNN)**  
This uses `train_and_detect.py`, which trains a custom CNN and then immediately starts webcam detection.

```bash
python train_and_detect.py
```

This script will:
- Load and preprocess your dataset from `C:\Users\Charan s\Desktop\Dataset`
- Train a CNN model (80% training, 20% testing)
- Save the trained model and label encoder
- Generate a training history plot
- Start real-time detection using your webcam

- **Option 2 – Recommended: Transfer learning + separate detection scripts**  
This uses MobileNetV2 for better accuracy and provides separate scripts for PC and Raspberry Pi.

1. **Train the model**:

```bash
python train.py
```

2. **Run real-time detection on a PC**:

```bash
python detect.py
```

3. **Run real-time detection on a Raspberry Pi**:

```bash
python raspberry_detect.py
```

`detect.py` and `raspberry_detect.py` both use the files produced by `train.py` (`object_detection_model.h5`, `label_encoder_classes.npy`, `label_mapping.npy`).

### Features

- **Flexible training options**:
  - Simple CNN (`train_and_detect.py`)
  - Transfer learning with MobileNetV2 (`train.py`)
- **Dataset-agnostic** as long as images are organized in class folders
- **Data preprocessing and augmentation** (rotation, shifts, flips) in `train.py`
- **Real-time detection** using webcam (PC) or camera (Raspberry Pi)
- **Training history visualization** saved to an image file

### Controls

- **Press `q`**: Quit the real-time detection window

### Output Files

- `object_detection_model.h5`: Trained model
- `label_encoder_classes.npy`: Encoded label classes
- `label_mapping.npy`: Mapping from numeric class indices to human-readable names (used by `detect.py` / `raspberry_detect.py`)

- `training_history.png`: Training accuracy and loss plots
