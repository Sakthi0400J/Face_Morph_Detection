import pickle
import numpy as np
import faiss
from deepface import DeepFace


try:
    with open('embeddings.pkl', 'rb') as f:
        database = pickle.load(f)
except FileNotFoundError:
    print("Error: 'embeddings.pkl' not found.")
    exit()


db_names = list(database.keys())
db_emb_matrix = np.vstack(list(database.values())).astype('float32')  # Shape: (N, 128), float32 for FAISS


def get_face_embedding(image_path):
    try:
        result = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        return np.array(result[0]['embedding']) if result else None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

image_path = r"C:\pradeep\IMG_7833.JPG"
query_embedding = get_face_embedding(image_path)

if query_embedding is None:
    print("Error: Could not generate embedding for the query image.")
    exit()


query_embedding = query_embedding.astype('float32').reshape(1, -1)  # Shape: (1, 128)


faiss.normalize_L2(db_emb_matrix)

faiss.normalize_L2(query_embedding)


dimension = 128
index = faiss.IndexFlatIP(dimension)
index.add(db_emb_matrix)


k = 10
distances, indices = index.search(query_embedding, k)


matches = []

print("Top 5 matching image names:")
for i in range(k):
    idx = indices[0][i]
    name = db_names[idx]
    matches.append(name)
    sim = distances[0][i]
    print(f"{i+1}. {name} (Similarity: {sim:.4f})")

with open('matches.txt', 'w') as f:
    for name in matches:
        f.write(name + '\n')

