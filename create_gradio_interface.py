import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load the trained models
custom_cnn = tf.keras.models.load_model('models/cnn/custom_cnn.keras')
vgg16 = tf.keras.models.load_model('models/pretrained/vgg16.keras')
resnet50 = tf.keras.models.load_model('models/pretrained/resnet50.keras')
inceptionv3 = tf.keras.models.load_model('models/pretrained/inceptionv3.keras')

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_expanded = np.expand_dims(img, axis=[0, -1])
    img_expanded_rgb = np.repeat(img_expanded, 3, axis=-1)
    return img_expanded, img_expanded_rgb

def majority_voting(*args):
    return (np.sum(args, axis=0) >= (len(args) / 2)).astype(int)

def predict(image):
    img, img_rgb = preprocess_image(image)
    custom_pred = custom_cnn.predict(img)[0][0]
    vgg16_pred = vgg16.predict(img_rgb)[0][0]
    resnet50_pred = resnet50.predict(img_rgb)[0][0]
    inceptionv3_pred = inceptionv3.predict(img_rgb)[0][0]
    
    # Binarize predictions
    custom_binary = (custom_pred > 0.5).astype(int)
    vgg16_binary = (vgg16_pred > 0.5).astype(int)
    resnet50_binary = (resnet50_pred > 0.5).astype(int)
    inceptionv3_binary = (inceptionv3_pred > 0.5).astype(int)
    
    # Ensemble prediction using majority voting
    ensemble_pred = majority_voting(custom_binary, vgg16_binary, resnet50_binary, inceptionv3_binary)
    
    labels = {
        "Custom CNN": "Cancer" if custom_binary else "No Cancer",
        "VGG16": "Cancer" if vgg16_binary else "No Cancer",
        "ResNet50": "Cancer" if resnet50_binary else "No Cancer",
        "InceptionV3": "Cancer" if inceptionv3_binary else "No Cancer",
        "Ensemble": "Cancer" if ensemble_pred else "No Cancer"
    }

    confidences = {
        "Custom CNN": float(custom_pred),
        "VGG16": float(vgg16_pred),
        "ResNet50": float(resnet50_pred),
        "InceptionV3": float(inceptionv3_pred),
        "Ensemble": float(np.mean([custom_pred, vgg16_pred, resnet50_pred, inceptionv3_pred]))
    }
    
    # Combine labels and confidences into a single dictionary
    results = {label: confidences[name] for name, label in labels.items()}
    
    # Return the ensemble label and results dictionary
    return labels["Ensemble"], results

interface = gr.Interface(fn=predict, inputs="image", outputs=["label", "json"])
interface.launch()

