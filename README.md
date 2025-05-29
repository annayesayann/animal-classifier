# Animal Image Classifier (90 Classes)

A deep learning project that classifies animals into 90 distinct classes using image data from a Kaggle dataset. This project achieves over 94% accuracy using a Convolutional Neural Network (CNN) model trained from scratch and enhanced with transfer learning.

---

## Dataset Overview

- **Source**: [Kaggle - 90 Animal Classes Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
- **Total Images**: 5,400
- **Classes**: 90 animal species
- **Images per class**: 60
- **Image Format**: JPEG
- **Image Size**: Resized to 224x224 for training

---

## Model Architecture

Two approaches were used:

### 1. Custom CNN
- 5 layered architecture
- 3×3 kernels with ReLU activation and batch normalization  
- Max pooling layers for downsampling 
- Fully connected dense layers
- Dropout regularization

### 2. Transfer Learning
- Pre-trained **MobileNetV2**, **ResNet50** , and **EfficientNet b0** as as feature extractors  

- Achieved faster convergence and higher accuracy

  > **Note**: During preprocessing, some visually similar animal classes were merged to reduce label noise and improve generalization, which can be found in main.ipynb.

---

## Key Results

| Model                 | Accuracy |
| --------------------- | -------- |
| CNN (custom)          | 44%      |
| ResNet50 (fine-tuned) | 88%      |
| MobileNet V2          | 88%      |
| EfficientNet B0       | **94%**  |

- Evaluation done using 80/20 train-test split.
- Data augmentation (flip, rotate, zoom, shift) applied during training to prevent overfitting.

## Contributors

- [@AnnaYesayann](https://github.com/annayesayann)

- [@MherBeginyan](https://github.com/mherbeginyan)

- [@SophieKhachatrian](https://github.com/SophieKhachatrian)

- [@VeronikaKhachatryan](https://github.com/VeronikkaKh)

- [@GorTadevosyan](https://github.com/gortadevosyan)

  

  
