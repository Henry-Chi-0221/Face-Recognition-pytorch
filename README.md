# Face Recognition based on pytorch
* Dataset : AFAD-LITE
* Link : https://afad-dataset.github.io/
* Github : https://github.com/afad-dataset/tarball-lite

* To solve data imbalance ,sample data are duplicate by 500 times and applied data augmentation.
* This project was implemented as two stage classifier : 
  * Haar cascade for face detection
  * CNN for classification
# Dependanceies
* OpenCV
* Pytorch
* Pillow
* Pandas
* numpy

# Usage
### Prepare for training
* Press C to capture sample image (x100)
```bash
python capture.py
```

* All sample images will be saved in ./capture 

### Training & Testing
```bash
python main.py
```
* The weight file will be saved as ./checkpoint.pth


# Demo
* Blue box for front-face detection
* Green box for target face detection
<p float="left">
  <img src="https://user-images.githubusercontent.com/48129098/117694577-e3157100-b1f1-11eb-8c23-4e2f3c24f0f2.png" width="300" />
  <img src="https://user-images.githubusercontent.com/48129098/117694585-e4df3480-b1f1-11eb-8d4e-1ebe072d4ead.png" width="300" /> 
</p>
