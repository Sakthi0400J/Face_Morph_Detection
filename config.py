# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PHOTO_DIR = os.path.join(DATA_DIR, "all_photos")

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
MATCHES_PATH = os.path.join(DATA_DIR, "matches.txt")

MODEL_PATH = os.path.join(BASE_DIR, "DeepFake-Detector", "cnn_model.h5")

# DeepFace settings
MODEL_NAME = "Facenet"
ENFORCE_DETECTION = True
TOP_K_MATCHES = 10