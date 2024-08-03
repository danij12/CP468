import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Define paths
processed_data_dir = 'data/processed'

# Load images and labels
def load_data(data_dir):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            label = 1 if 'cancer' in filename else 0  # Assuming filenames have labels in them
            labels.append(label)
    return np.array(images), np.array(labels)

X, y = load_data(processed_data_dir)

# Normalize the images
X = X / 255.0
X = X.reshape(-1, 224, 224, 1)  # Reshape to add channel dimension

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save the splits
np.save('data/splits/train_images.npy', X_train)
np.save('data/splits/train_labels.npy', y_train)
np.save('data/splits/val_images.npy', X_val)
np.save('data/splits/val_labels.npy', y_val)
np.save('data/splits/test_images.npy', X_test)
np.save('data/splits/test_labels.npy', y_test)

print("Data loading and splitting complete.")
