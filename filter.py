import pickle
import numpy as np
import faiss
from deepface import DeepFace

# Step 1: Load embeddings from 'embeddings.pkl'
# Assumes the file is a dict {filename: np.array(128)}
try:
    with open('embeddings.pkl', 'rb') as f:
        database = pickle.load(f)
except FileNotFoundError:
    print("Error: 'embeddings.pkl' not found.")
    exit()

# Step 2: Prepare the database embeddings
# Extract names and stack into a matrix
db_names = list(database.keys())
db_emb_matrix = np.vstack(list(database.values())).astype('float32')  # Shape: (N, 128), float32 for FAISS

# Step 3: Get query embedding from image
def get_face_embedding(image_path):
    try:
        result = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        return np.array(result[0]['embedding']) if result else None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Usage: Replace with your image path
image_path = r"C:\pradeep\IMG_7833.JPG"
query_embedding = get_face_embedding(image_path)

if query_embedding is None:
    print("Error: Could not generate embedding for the query image.")
    exit()

# Reshape to (1, 128) and convert to float32
query_embedding = query_embedding.astype('float32').reshape(1, -1)  # Shape: (1, 128)

# Step 4: Normalize vectors for cosine similarity (L2 norm)
# Normalize database (in-place)
faiss.normalize_L2(db_emb_matrix)
# Normalize query (in-place)
faiss.normalize_L2(query_embedding)

# Step 5: Build FAISS index for Inner Product (cosine similarity)
dimension = 128
index = faiss.IndexFlatIP(dimension)
index.add(db_emb_matrix)

# Step 6: Search for top 5 matches
k = 10
distances, indices = index.search(query_embedding, k)


matches = []
# Step 7: Extract and print the top 5 image names
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
