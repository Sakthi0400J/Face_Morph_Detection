import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import config


def load_matches():
    if not os.path.exists(config.MATCHES_PATH):
        print("matches.txt not found.")
        exit()

    with open(config.MATCHES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


def predict_deepfake(model, img_path):

    img = image.load_img(img_path, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)[0][0]

    if pred < 0.5:
        result = f"REAL ({1-pred:.2f})"
    else:
        result = f"FAKE ({pred:.2f})"

    print(f"{os.path.basename(img_path)} → {result}")


def main():

    print("Loading CNN model...")
    model = load_model(config.MODEL_PATH)

    images = load_matches()

    print("\nRunning Deepfake Detection:\n")

    for img in images:
        if os.path.exists(img):
            predict_deepfake(model, img)
        else:
            print("Missing:", img)


if __name__ == "__main__":
    main()