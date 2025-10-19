import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from mtcnn import MTCNN

# Load the trained model
model = load_model('cnn_model.h5')

# Function to detect and crop the face using MTCNN
def detect_and_crop_face(img_path):
    img = cv2.imread(img_path)
    detector = MTCNN()
    results = detector.detect_faces(img)
    
    if results:
        bounding_box = results[0]['box']
        x, y, width, height = bounding_box
        face = img[y:y+height, x:x+width]
        face = cv2.resize(face, (128, 128))  # Resize to the target size
        return face
    else:
        # If no face is detected, return the original resized image
        return cv2.resize(img, (128, 128))

# Function to preprocess the cropped face
def preprocess_face(face):
    img_array = image.img_to_array(face)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Main function to predict if an image is real or fake
def predict_real_or_fake(img_path):
    # Detect and preprocess the face in the provided image
    face = detect_and_crop_face(img_path)
    processed_image = preprocess_face(face)

    # Predict the class of the image
    prediction = model.predict(processed_image)

    # Print the result
    result = 'Real' if prediction[0][0] < 0.5 else 'Fake'
    print(f"Image: {img_path} - Predicted as: {result}")

# Example usage:
image_path = input("Enter your image path: ")  # Replace with your image path
predict_real_or_fake(image_path)
