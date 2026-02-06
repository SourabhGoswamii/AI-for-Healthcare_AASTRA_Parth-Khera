
import os
import numpy as np
import nibabel as nib
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


DATA_DIR = "../MRI_TASK1/processed_dataset"
IMG_SIZE = 128
SLICE_COUNT = 10
EPOCHS = 15
BATCH_SIZE = 16

X, y = [], []

def load_subject(path, label):
    img = nib.load(path).get_fdata()
    mid = img.shape[2] // 2

    for i in range(mid - SLICE_COUNT // 2, mid + SLICE_COUNT // 2):
        slice_img = img[:, :, i]
        slice_img = cv2.resize(slice_img, (IMG_SIZE, IMG_SIZE))
        X.append(slice_img)
        y.append(label)

print("\n Loading MRI data...")

for cls, label in [("CN", 0), ("AD", 1)]:
    folder = os.path.join(DATA_DIR, cls)
    for file in os.listdir(folder):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            load_subject(os.path.join(folder, file), label)

print(f" Total MRI slices loaded: {len(X)}")


X = np.array(X, dtype=np.float32)
X = X[..., np.newaxis]   # (N,128,128,1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train data:", X_train.shape)
print("Test data :", X_test.shape)


model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\n Training started...\n")

model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)


print("\n Evaluating model...")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n TEST ACCURACY: {test_accuracy * 100:.2f}%")

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

print("\n CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\n CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=["CN", "AD"]))

auc = roc_auc_score(y_test, y_prob)
print(f" ROC-AUC SCORE: {auc:.3f}")


os.makedirs("models", exist_ok=True)
model.save("models/task2_cn_vs_ad_model.h5")

print("\n TASK 2 COMPLETED SUCCESSFULLY")
print(" Model saved at: MRI_TASK2/models/task2_cn_vs_ad_model.h5")
