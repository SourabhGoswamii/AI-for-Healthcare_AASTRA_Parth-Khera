# AI-for-Healthcare_AASTRA_Parth-Khera

## Overview
This project implements an end-to-end healthcare AI pipeline for neurological disorder detection using brain MRI scans.  
The system covers **dataset preprocessing**, **binary classification**, and **multi-class classification**.

All experiments were conducted on a **Linux GPU server accessed via VPN and SSH**.

---

## Dataset
- **MRI:** Brain MRI scans in DICOM (.dcm) format  
- **Labels:** CSV file with subject-level diagnoses:
  - CN (Cognitively Normal)
  - MCI (Mild Cognitive Impairment)
  - AD (Alzheimer’s Disease)

The CSV file is the **single source of ground truth**.

---

## Task 1: Dataset Preprocessing

**Objective:** Ensure data integrity, standardization, and reproducibility.

**Key Steps:**
- Automatic data extraction
- Recursive DICOM loading
- 3D MRI volume reconstruction
- Noise removal and global intensity normalization
- Central slice extraction (3D → 2D)
- Label encoding
- Leakage-free train/test split

**Compliance:**
- No data augmentation
- No label modification
- No sample addition/removal
- No class rebalancing
- No manual region selection

---

## Task 2: Binary Classification (CN vs AD)

**Objective:** Detect Alzheimer’s disease from healthy controls using MRI data.

**Method:**
- Central MRI slices resized to 128×128
- 2D CNN with sigmoid output
- Loss: Binary Crossentropy
- Optimizer: Adam
- Early stopping

**Evaluation:**
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC–AUC

**Result:**  
Achieved **>91% test accuracy**, meeting the task requirement.

---

## Task 3: Multi-Class Classification (CN vs MCI vs AD)

**Objective:** Distinguish between CN, MCI, and AD in a clinically realistic setting.

**Method:**
- Slice-based learning with subject-level aggregation
- 2D CNN with softmax output
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Early stopping

**Evaluation:**
- Overall and per-class metrics
- Confusion matrix

**Result:**  
Achieved **>55% accuracy**, exceeding the required threshold.

---

## Summary
This project delivers a compliant and reproducible MRI-based AI system capable of binary and multi-class neurological disorder classification, suitable for real-world screening scenarios.
