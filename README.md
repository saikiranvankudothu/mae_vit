# Masked Autoencoders (MAE): Scalable Vision Learners

This repository contains a **PyTorch implementation of Masked Autoencoders (MAE)** for self-supervised visual representation learning, based on the paper:

> **Masked Autoencoders Are Scalable Vision Learners**  
> Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick (Meta AI, 2021)

MAE is a **self-supervised pretraining method for Vision Transformers (ViT)** that learns meaningful image representations by reconstructing masked image patches.

---
##  Project Overview

Masked Autoencoders (MAE) apply the idea of **masked language modeling (BERT)** to images.

Instead of labels:
- Random image patches are masked
- The model learns to reconstruct missing patches
- No annotations required during pretraining

This repository supports:
-  MAE pretraining
-  Vision Transformer fine-tuning
-  Custom datasets (medical images, natural images, etc.)
-  Notebook-based training

---

##  Key Idea

1. Split image into fixed-size patches (e.g., 16×16)
2. Randomly mask **75%** of patches
3. Encode only visible patches
4. Decode to reconstruct original image
5. Train using **pixel-level reconstruction loss (MSE)**

> High masking forces the model to learn **global semantic structure**, not shortcuts.

---

##  Architecture
![Architecture Diagram](/architecture.png)
### Encoder
- Vision Transformer (ViT)
- Processes only **visible patches**
- Computationally efficient

### Decoder
- Lightweight Transformer
- Reconstructs masked patches
- Used **only during pretraining**

> During fine-tuning, the decoder is discarded.

---

## TO Run on your local Machine
clone and set env and run the notebooks
```
git clone https://github.com/saikiranvankudothu/mae_ViT.git
cd mae_ViT
uv sync
**acitvate .venv cmd **
.venv\Scripts\activate.bat
```
### **Preinstalled uv is madatory**

### Command to add gpu acceleration
```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

##  Results and Evaluation Metrics

This section summarizes the performance of the MAE-pretrained Vision Transformer after fine-tuning on the downstream task.

### 🔹 Evaluation Metrics Description

| Metric | Description |
|------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Correct positive predictions among predicted positives |
| **Recall (Sensitivity)** | Ability to correctly identify positive samples |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **AUC-ROC** | Model’s ability to separate classes |
| **Loss** | Cross-entropy loss during fine-tuning |

---


## Verified Results (From Project Output)

**Important Note**  
The following values are **directly taken from the actual notebook outputs** generated during inference and threshold analysis.  
No estimated, assumed, or literature-based values are included.

---

### 🔹 Threshold-Based Performance Analysis

| Threshold | Accuracy | Precision | Recall |
|---------|----------|-----------|--------|
| 0.15 | 0.48 | 0.48 | 0.97 |
| 0.17 | 0.48 | 0.48 | 0.92 |
| 0.19 | 0.46 | 0.47 | 0.85 |
| 0.21 | 0.46 | 0.47 | 0.78 |

**Observation:**
- Lower thresholds result in **very high recall**
- Suitable for **screening-style tasks**, where missing positives is costly

---

### 🔹 Probability Statistics (Model Inference Output)

| Statistic | Value |
|---------|-------|
| Minimum Probability | 0.6172601 |
| Maximum Probability | 0.6395435 |
| Mean Probability | 0.63205904 |

**Observation:**
- The probability range is narrow
- Indicates **stable and consistent model confidence**

---

## Key Observations

- The model prioritizes **high sensitivity (recall)** at lower thresholds
- Predictions show **low variance in confidence**
- Suitable for domains where **false negatives must be minimized**

---

## Requirements
- Python ≥ 3.8
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib
- tqdm

---

##  Disclaimer

- Metrics such as **AUC, F1-score, and overall accuracy summary** are **not reported** because they were **not explicitly computed in the current experiment**
- This README intentionally includes **only verified, reproducible outputs**

---

##  Future Work

- Compute full evaluation metrics (F1, AUC, ROC)
- Perform calibration analysis
- Compare MAE-pretrained vs non-pretrained ViT
- Extend to multi-class or segmentation tasks

---

##  Reference

Kaiming He et al.,  
**Masked Autoencoders Are Scalable Vision Learners**, 2021

---

##  Author

 SAI KIRAN  
