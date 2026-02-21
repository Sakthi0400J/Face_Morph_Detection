import pickle
import numpy as np
import faiss
from deepface import DeepFace
import config
import os


def load_database():
    if not os.path.exists(config.EMBEDDINGS_PATH):
        print("Embeddings not found.")
        exit()

    with open(config.EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)


def get_face_embedding(image_path):
    result = DeepFace.represent(
        img_path=image_path,
        model_name=config.MODEL_NAME,
        enforce_detection=config.ENFORCE_DETECTION
    )
    return np.array(result[0]["embedding"])


def main():

    database = load_database()

    db_names = list(database.keys())
    db_matrix = np.vstack(list(database.values())).astype("float32")

    query_image = input("Enter query image path: ").strip()
    query_embedding = get_face_embedding(query_image).astype("float32").reshape(1, -1)

    # normalize safely
    db_matrix = db_matrix.copy()
    faiss.normalize_L2(db_matrix)
    faiss.normalize_L2(query_embedding)

    dimension = db_matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(db_matrix)

    k = min(config.TOP_K_MATCHES, len(db_names))
    distances, indices = index.search(query_embedding, k)

    matches = []

    print("\nTop Matches:")
    for i in range(k):
        name = db_names[indices[0][i]]
        sim = distances[0][i]
        print(f"{i+1}. {name} ({sim:.4f})")
        matches.append(os.path.join(config.PHOTO_DIR, name))

    with open(config.MATCHES_PATH, "w") as f:
        for m in matches:
            f.write(m + "\n")


if __name__ == "__main__":
    main()