import tensorflow as tf

matched_names = []

with open('matches.txt', 'r') as f:
    for line in f:
        name = line.strip()  
        if name:  
            matched_names.append(name)
print(matched_names)

from keras.preprocessing import image
import tensorflow as tf

import numpy as np



import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Path to the downloaded model (update if different)
model_path = r'C:\Users\admin\sample_face_matrix\DeepFake-Detector\cnn_model.h5'  # Or wherever you saved it

# List of your images (from your earlier output)
image_list = ['IMG_7833.JPG', 'IMG_7835.JPG', 'IMG_7834.JPG', 'IMG_7836.JPG', 'IMG_7832.JPG', 'IMG_7837.JPG', 'IMG_7831.JPG', 'IMG_7827.JPG', 'IMG_7825.JPG', 'IMG_7842.JPG']
folder_path = r'C:\Users\admin\sample_face_matrix\all_photos'  # Folder with your images

# Load the model
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to predict on a single image
def predict_deepfake(img_path):
    try:
        # Load and preprocess the image (resize to 128x128 as required)
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0  # Normalize to 0-1
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img_array)
        prob = prediction[0][0]  # Probability of being "Fake" (0-1)
        
        # Interpret: < 0.5 = Real, >= 0.5 = Fake
        if prob < 0.5:
            result = f"Real (confidence: {1 - prob:.2f})"
        else:
            result = f"Fake/Deepfake (confidence: {prob:.2f})"
        
        print(f"{os.path.basename(img_path)}: {result}")
        return result
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return "Error"

# Process all images
print("\nRunning deepfake detection on your images:")
for img in image_list:
    full_path = os.path.join(folder_path, img)
    if os.path.exists(full_path):
        predict_deepfake(full_path)
    else:
        print(f"Image not found: {full_path}")
