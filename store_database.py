import os
import pickle  
from deepface import DeepFace
import numpy as np

def get_face_embedding(image_path, model_name="Facenet", enforce_detection=True):
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            enforce_detection=enforce_detection
        )
        if result:
            return np.array(result[0]['embedding'])
        else:
            print(f"No face detected in {image_path}")
            return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_folder_and_save_embeddings(folder_path, output_file="embeddings.pkl"):
    
    embeddings = {}  
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            embedding = get_face_embedding(image_path)
            if embedding is not None:
                embeddings[filename] = embedding
                print(f"  Embedding generated for {filename} (shape: {embedding.shape})")
            else:
                print(f"  Skipped {filename} (no valid embedding)")
    
    
    if embeddings:
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"\nEmbeddings saved to {output_file}")
        print(f"Total embeddings: {len(embeddings)}")
    else:
        print("No embeddings generated. Check your images and folder.")


folder_path = r"C:\Users\admin\sample_face_matrix\all_photos" 
output_file = r"C:\Users\admin\sample_face_matrix\embeddings.pkl"    
process_folder_and_save_embeddings(folder_path, output_file)

import pickle
with open('embeddings.pkl', 'rb') as f:
    loaded_embeddings = pickle.load(f)

print(loaded_embeddings)  
