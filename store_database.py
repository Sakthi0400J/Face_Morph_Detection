import os
import pickle
import numpy as np
from deepface import DeepFace
import config


def get_face_embedding(image_path):
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=config.MODEL_NAME,
            enforce_detection=config.ENFORCE_DETECTION
        )
        return np.array(result[0]["embedding"]) if result else None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_folder():
    embeddings = {}
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    if not os.path.isdir(config.PHOTO_DIR):
        print("Photo folder not found.")
        return

    for filename in os.listdir(config.PHOTO_DIR):
        if filename.lower().endswith(extensions):

            path = os.path.join(config.PHOTO_DIR, filename)
            print(f"Processing {filename}")

            emb = get_face_embedding(path)

            if emb is not None:
                embeddings[filename] = emb
                print(f"✓ Stored ({emb.shape})")

    if embeddings:
        with open(config.EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)

        print(f"\nSaved {len(embeddings)} embeddings.")
    else:
        print("No embeddings created.")


if __name__ == "__main__":
    process_folder()