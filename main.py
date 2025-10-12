import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_utils import build_model, save_model

TRAIN_DIR = r"C:\Users\legit\OneDrive\Desktop\SKILLCRAFT_TASK4\train\train"
TEST_DIR = r"C:\Users\legit\OneDrive\Desktop\SKILLCRAFT_TASK4\test\test"
IMG_SIZE = (64,64)

def load_data(data_dir, img_size=IMG_SIZE):
    X, y, gesture_labels = [], [], []

    for gesture_folder in sorted(os.listdir(data_dir)):
        gesture_path = os.path.join(data_dir, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue
        gesture_labels.append(gesture_folder)
        print(f"Loading images from: {gesture_path}")

        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: cannot read {img_path}")
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(gesture_folder)

        print(f"Loaded {len(X)} images for gesture '{gesture_folder}'")

    if len(X) == 0:
        raise ValueError(f"No images loaded from {data_dir}! Check folder structure.")

    X = np.array(X, dtype='float32') / 255.0
    X = X.reshape(-1, img_size[0], img_size[1], 1)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    return X, y_onehot, le.classes_

X_train, y_train, gesture_labels = load_data(TRAIN_DIR)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.1
)

train_gen = datagen.flow(X_train, y_train, batch_size=64, subset='training')
val_gen = datagen.flow(X_train, y_train, batch_size=64, subset='validation')

model = build_model(input_shape=(64,64,1), num_classes=len(gesture_labels))
model.fit(train_gen, validation_data=val_gen, epochs=20)

save_model(model)
np.save('gesture_labels.npy', gesture_labels)
print("Training done. Model saved as hand_gesture_model.h5")

X_test, y_test, _ = load_data(TEST_DIR)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")
