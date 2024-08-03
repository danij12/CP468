import argparse
from scripts.preprocess_data import preprocess_dicom
from scripts.create_gradio_interface import interface

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description='AI Lung Cancer Detection')
    parser.add_argument('--mode', type=str, required=True, choices=['preprocess', 'train', 'interface'], help='Mode to run the script: preprocess, train, interface')
    args = parser.parse_args()

    if args.mode == 'preprocess':
        # Define paths
        raw_data_dir = 'data/raw/'
        processed_data_dir = 'data/processed'
        metadata_path = 'data/Raw/metadata.csv'

        # Create processed_data_dir if it doesn't exist
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)

        # Load metadata
        metadata = pd.read_csv(metadata_path)

        # Run preprocessing
        preprocess_dicom(metadata, raw_data_dir, processed_data_dir)

        print("Preprocessing complete.")

    elif args.mode == 'train':
        # Load images and labels
        def load_data(data_dir):
            images = []
            labels = []
            for filename in os.listdir(data_dir):
                if filename.endswith('.png'):
                    img_path = os.path.join(data_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    images.append(img)
                    label = 1  # Since all images contain cancer
                    labels.append(label)
            return np.array(images), np.array(labels)

        processed_data_dir = 'data/processed'
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
        # Add training code here if needed

    elif args.mode == 'interface':
        # Run the Gradio interface
        interface.launch()

if __name__ == "__main__":
    main()
