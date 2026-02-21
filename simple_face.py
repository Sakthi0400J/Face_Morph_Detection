# simple_face.py
from deepface import DeepFace
import numpy as np
import config


def get_face_embedding(image_path):
    result = DeepFace.represent(
        img_path=image_path,
        model_name=config.MODEL_NAME,
        enforce_detection=config.ENFORCE_DETECTION
    )
    return np.array(result[0]["embedding"])


img = input("Enter image path: ")
embedding = get_face_embedding(img)

print("Embedding shape:", embedding.shape)