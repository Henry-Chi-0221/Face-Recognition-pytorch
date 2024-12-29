# Face Recognition using PyTorch

## Overview

This project demonstrates face recognition using PyTorch. It leverages OpenCV's Haar Cascade for face detection and a Convolutional Neural Network (CNN) for classification. The Haar Cascade identifies faces in real-time video streams, which are then cropped and resized to 224x224 pixels. These processed images are input into a ResNet18-based CNN for binary classification, distinguishing between positive and negative samples.

## Dataset

- **Dataset**: AFAD-LITE
- **Link**: [AFAD Dataset](https://afad-dataset.github.io/)
- **GitHub**: [AFAD Repository](https://github.com/afad-dataset/tarball-lite)

## Key Features

- **Data Imbalance Handling**: To address class imbalance, samples are duplicated 500 times, and data augmentation techniques such as horizontal flipping, random rotation, and vertical flipping are applied to improve model generalization.
- **Two-Stage Classification:**
  1. **Haar Cascade for frontal face detection:**
     - Implements OpenCV's pre-trained 'haarcascade_frontalface_default.xml' for face detection.
     - Uses a scale factor of 1.2 and a minimum neighbor count of 3.
     - Filters out detected faces smaller than 100x100 pixels to ensure input quality.
  2. **CNN for classification:**
     - Employs a ResNet18 pre-trained model modified with a fully connected layer (512 inputs, 2 outputs).
     - Optimized with Adam optimizer and CrossEntropy loss.
     - Trained for 2 epochs with a batch size of 4 and a learning rate of 0.0005.

## Dependencies

- OpenCV
- PyTorch
- Pillow
- Pandas
- NumPy

## Usage

### Data Preparation

- Press `C` to capture sample images (100 samples).

```bash
python capture.py
```

- Captured images are saved in the `./capture` directory.

### Training and Testing

- Train and test the model.

```bash
python main.py
```

- Model weights are saved as `./checkpoint.pth`.

## Scripts

### 1. capture.py
- Captures sample images using OpenCV's Haar Cascade face detector.
- Detects faces in real-time and saves cropped images of size 224x224 pixels.
- Press 'C' to capture samples and 'Q' to quit.

### 2. demo.py
- Performs real-time face recognition using the trained model.
- Highlights detected faces with bounding boxes and classifies them as positive or negative.
- Stabilizes predictions with a counter requiring 5 consecutive positive detections.
- Press 'C' to save the frame and 'Q' to quit.

### 3. dataset.py
- Prepares the dataset and handles data augmentation.
- Splits data into training (70%) and testing (30%).
- Handles data imbalance by duplicating samples and applying transformations.

### 4. main.py
- Trains and tests the model.
- Uses ResNet18 as the CNN backbone and saves checkpoints after training.

### 5. util.py
- Provides utilities for data conversion between OpenCV, PIL, and PyTorch tensor formats.
- Ensures consistent preprocessing for input data.

## Notes

Ensure dependencies are installed before running the scripts:

```bash
pip install -r requirements.txt
```
