# Skin_Cancer_Classifier
# Skin Cancer Classification using Deep Learning (HAM10000)

This project implements a deep learning pipeline for **multi-class skin lesion classification**
using the **HAM10000 dataset**. The model uses **transfer learning with ResNet50** and includes
evaluation metrics and **Grad-CAM visual explanations**.

---

## Dataset
- **HAM10000 (Human Against Machine with 10000 training images)**
- 7 skin lesion classes:
  - Melanocytic nevi (nv)
  - Melanoma (mel)
  - Basal cell carcinoma (bcc)
  - Actinic keratoses (akiec)
  - Benign keratosis-like lesions (bkl)
  - Vascular lesions (vasc)
  - Dermatofibroma (df)

> Dataset is not included in this repository due to size.

---

## Project Workflow
1. Load metadata and preprocess image filenames
2. Analyze class imbalance
3. Stratified train / validation / test split
4. Image preprocessing and augmentation
5. Transfer learning with ResNet50
6. Class-weighted training
7. Model evaluation:
   - Confusion matrix
   - Classification report
   - ROC–AUC curves
8. Model saving
9. Explainability using Grad-CAM

---

## Model Architecture
- Base model: **ResNet50 (ImageNet pretrained)**
- Custom classifier head:
  - Global Average Pooling
  - Dense (ReLU)
  - Dropout
  - Softmax output (7 classes)

---

## Results & Observations
- The dataset is **highly imbalanced**, which causes the model to bias toward majority classes.
- Some classes show low precision/recall.
- Grad-CAM visualizations highlight regions used by the model for prediction.
- Due to severe class imbalance in HAM10000, the model shows bias toward majority classes, highlighting the need for advanced imbalance-handling techniques.

---

## Requirements
- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib / Seaborn
- scikit-learn
- OpenCV

---

## How to Run
1. Download the HAM10000 dataset
2. Upload notebook to Google Colab
3. Update dataset paths if required
4. Run all cells sequentially

---

## Notes
This project demonstrates:
- Transfer learning
- Handling class imbalance
- Model evaluation
- Model explainability in medical imaging

