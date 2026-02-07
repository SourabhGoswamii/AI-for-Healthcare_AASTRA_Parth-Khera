# =====================================================
# TASK 1: MRI DATA PREPARATION PIPELINE (FINAL FINAL)
# Handles ADNI MRI with MPRAGE_REPE, MPRAGE_SENS, etc.
# =====================================================

import os
import zipfile
import numpy as np
import pandas as pd
import pydicom
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =====================================================
# MODIFY ONLY IF NAMES ARE DIFFERENT
# =====================================================
ZIP_FILE_NAME = "MRI.zip"
CSV_FILE_NAME = "MRI_metadata.csv"
MRI_FOLDER_NAME = "MRI"
# =====================================================


# =====================================================
# STEP 1: UNZIP MRI DATA
# =====================================================
if not os.path.exists(MRI_FOLDER_NAME):
    print("Unzipping MRI data...")
    with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
        zip_ref.extractall(".")
else:
    print("MRI folder already exists. Skipping unzip.")


# =====================================================
# STEP 2: READ CSV METADATA
# =====================================================
print("Reading CSV metadata...")
df = pd.read_csv(CSV_FILE_NAME)

SUBJECT_COLUMN = df.columns[0]   # e.g. 002_S_0413
LABEL_COLUMN = df.columns[1]     # CN / MCI / AD

df = df[[SUBJECT_COLUMN, LABEL_COLUMN]]
df = df.groupby(SUBJECT_COLUMN).first().reset_index()

print("Total unique patients:", len(df))


# =====================================================
# STEP 3: RECURSIVE DICOM LOADER (KEY STEP)
# =====================================================
def load_brain(folder_path):
    slices = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".dcm"):
                try:
                    dcm = pydicom.dcmread(os.path.join(root, file))
                    slices.append(dcm.pixel_array)
                except:
                    continue

    if len(slices) == 0:
        return None

    return np.stack(slices, axis=-1)


# =====================================================
# STEP 4: SIMPLE PREPROCESSING
# =====================================================
def skull_strip(brain):
    brain = brain.copy()
    brain[brain < brain.mean()] = 0
    return brain

def normalize(brain):
    if brain.max() == brain.min():
        return brain
    return (brain - brain.min()) / (brain.max() - brain.min())

def get_slices(brain, size=128, count=10):
    depth = brain.shape[2]
    mid = depth // 2
    half = count // 2

    images = []
    for i in range(mid - half, mid + half):
        if 0 <= i < depth:
            img = cv2.resize(brain[:, :, i], (size, size))
            images.append(img)
    return images


# =====================================================
# STEP 5: BUILD DATASET
# =====================================================
X, y = [], []

print("Processing MRI images...")

for _, row in df.iterrows():
    subject_id = str(row[SUBJECT_COLUMN])
    label = row[LABEL_COLUMN]

    patient_folder = os.path.join(MRI_FOLDER_NAME, subject_id)

    if not os.path.exists(patient_folder):
        continue

    brain = load_brain(patient_folder)
    if brain is None:
        continue

    brain = skull_strip(brain)
    brain = normalize(brain)

    slices = get_slices(brain)

    for img in slices:
        X.append(img)
        y.append(label)

print("Total 2D images created:", len(X))


# =====================================================
# STEP 6: LABEL ENCODING
# =====================================================
le = LabelEncoder()
y = le.fit_transform(y)

print("Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))


# =====================================================
# STEP 7: FINAL FORMAT + TRAIN/TEST SPLIT
# =====================================================
X = np.array(X)
X = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\n✅ TASK 1 COMPLETED SUCCESSFULLY")
