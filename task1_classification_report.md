\# Task 1 — Pneumonia Classification



\## 1. Dataset Analysis



\### 1.1 Dataset Description



This project uses the \*\*PneumoniaMNIST\*\* dataset from the MedMNIST benchmark collection. PneumoniaMNIST is derived from pediatric chest X-ray images and is designed for binary classification:



\- \*\*Class 0\*\*: Normal

\- \*\*Class 1\*\*: Pneumonia



The dataset contains grayscale chest X-ray images resized to 28×28 pixels in the original MedMNIST format. For deep learning experiments, images were resized to match the input requirements of pretrained CNN and transformer models.



\### 1.2 Dataset Split



The dataset is divided into:



\- Training set: 4708

\- Validation set: 524

\- Test set: 624

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Total: 5856



The predefined splits provided by MedMNIST were used to ensure fair comparison and reproducibility.



\### 1.3 Class Distribution



Class balance is critical in medical imaging tasks because imbalance may bias models toward the majority class.

The PneumoniaMNIST dataset is relatively balanced, though slightly skewed toward pneumonia cases. This makes recall (sensitivity) particularly important, as failing to detect pneumonia (false negatives) could have serious clinical implications.



./data\_analysis/class\_distribution.png





\### 1.4 Challenges of the Dataset



Several challenges exist when working with PneumoniaMNIST:



\- Low resolution (28×28 original images)

\- Subtle radiographic patterns distinguishing normal vs pneumonia

\- Risk of overfitting due to limited spatial information

\- Medical domain complexity requiring robust feature extraction



These characteristics justify the use of transfer learning from pretrained models to improve feature representation.

