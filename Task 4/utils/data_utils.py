import os
import cv2
import numpy as np

def load_dataset(base_path, img_size=64, max_per_class=50):
    X, y = [], []
    categories = sorted(os.listdir(base_path))

    for label, gesture_folder in enumerate(categories):
        gesture_path = os.path.join(base_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        images = os.listdir(gesture_path)[:max_per_class]  # limit to avoid OOM
        for img_name in images:
            img_path = os.path.join(gesture_path, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label)
            except:
                print("❌ Error loading image:", img_path)

    X = np.array(X) / 255.0  # Normalize
    X = X.reshape(len(X), -1)  # Flatten
    y = np.array(y)

    return X, y
