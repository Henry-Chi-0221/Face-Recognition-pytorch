# Face Recognition based on pytorch
* Dataset : AFAD-LITE
* Link : https://afad-dataset.github.io/
* Github : https://github.com/afad-dataset/tarball-lite

* To solve data imbalance ,sample data are duplicate by 500 times and applied data augmentation
# Demo
* Blue box for front-face detection
* Green box for target face detection

# Usage
* Prepare for training
* Press C to capture sample image (x100)
```bash
python capture.py
```

* All sample images will be saved in ./capture 

* Training & Testing
```bash
python main.py
```
* The weight file will be saved as ./checkpoint.pth

