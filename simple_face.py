from deepface import DeepFace
import numpy as np

def get_face_embedding(image_path):
    try:
        result = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        return np.array(result[0]['embedding']) if result else None
    except Exception as e:
        print(f"Error: {e}")
        return None


embedding = get_face_embedding(r"C:\pradeep\IMG_7833.JPG")
if embedding is not None:
    print(f"Embedding shape: {embedding.shape}") 
    print(embedding)

    
