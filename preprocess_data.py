import os
import pydicom
import numpy as np
import cv2
import pandas as pd

def preprocess_dicom(metadata, input_folder, output_folder, img_size=(224, 224)):
    for index, row in metadata.iterrows():
        # Construct the directory path using raw strings
        dir_path = os.path.normpath(os.path.join(input_folder, row['File Location'].replace('.\\', '').replace('\\', '/')))
        
        print(f"Constructed directory path: {dir_path}")
        print(f"Exists: {os.path.exists(dir_path)}, Is Directory: {os.path.isdir(dir_path)}")

        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            if row['Modality'] == 'CT':
                print(f"Processing directory: {dir_path}")
                for filename in os.listdir(dir_path):
                    if filename.endswith('.dcm'):
                        file_path = os.path.join(dir_path, filename)
                        try:
                            dicom = pydicom.dcmread(file_path)
                            img = dicom.pixel_array
                            img = cv2.resize(img, img_size)
                            img = img / np.max(img)  # Normalize
                            # Generate a unique filename
                            output_filename = f"{row['Subject ID']}_{index}_{filename.replace('.dcm', '.png')}"
                            output_path = os.path.normpath(os.path.join(output_folder, output_filename))
                            print(f"Saving processed image to: {output_path}")
                            cv2.imwrite(output_path, (img * 255).astype(np.uint8))
                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")
            else:
                print(f"Skipping non-CT modality directory: {dir_path}")
        else:
            print(f"Directory does not exist or is not a directory: {dir_path}")

    print("Preprocessing complete.")

