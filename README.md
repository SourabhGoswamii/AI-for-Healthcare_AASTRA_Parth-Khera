# AI-for-Healthcare_AASTRA_Parth-Khera
# 🧠 MRI Dataset Preprocessing – Task 1

## Overview
This project implements **Task-1 preprocessing** for brain MRI–based neurological disorder detection.  
All work was executed on a **Linux GPU server accessed securely via VPN and SSH**.

The objective is **data standardization and integrity**, not model accuracy.

---

## Dataset
- **MRI:** DICOM (.dcm) brain scans in nested folders  
- **Labels:** CSV file with patient IDs and diagnoses (CN, MCI, AD)

The CSV file is the **only source of ground truth**.

---

## Pipeline Summary
1. Automatically extracts MRI data when required  
2. Loads and validates labels from CSV  
3. Recursively detects and reads all DICOM slices  
4. Reconstructs 3D brain MRI volumes  
5. Applies basic noise removal and intensity normalization  
6. Extracts informative middle slices (3D → 2D)  
7. Assigns labels from CSV and encodes them numerically  
8. Prepares train/test datasets without leakage  

---

## Compliance
- No data augmentation  
- No label changes  
- No sample addition or removal  
- No class rebalancing  
- No manual region selection  

All preprocessing is **uniform, deterministic, and reproducible**.

---

## One-Line Summary
This pipeline converts raw MRI DICOM data and clinical labels into a clean, AI-ready dataset using a secure Linux GPU environment, without manipulating the original data.

---
