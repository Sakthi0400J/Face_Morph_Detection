---
license: mit
language:
- en
metrics:
- accuracy
library_name: tf-keras
pipeline_tag: image-classification
---
---

# CNN-Based Deepfake Detection Model

This repository contains a Convolutional Neural Network (CNN)-based model fine-tuned for deepfake detection. The model has been trained to classify images as either "real" or "fake" (deepfake) using a custom dataset of processed images.

## Model Overview

This model is a custom CNN architecture built specifically for deepfake detection. It has been designed to efficiently distinguish between real and fake images through a series of convolutional and pooling layers, followed by fully connected layers for classification.

### Key Features:
- **Model Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 128x128 pixels
- **Number of Classes:** 2 (Real, Fake)
- **Activation Function:** ReLU in hidden layers, Sigmoid for binary classification
- **Regularization:** L2 regularization and Dropout layers to prevent overfitting
- **Optimizer:** Adam with a learning rate of 0.0001
- **Training Epochs:** 100 epochs (with early stopping based on validation loss)

## Training Details

The model was trained on a custom dataset of real and deepfake images, using data augmentation techniques to improve generalization. The training process involved the following components:

- **Data Augmentation:** Random rotations, shifts, flips, and brightness adjustments.
- **Loss Function:** Binary Cross-Entropy Loss
- **Optimizer:** Adam with a learning rate of 0.0001
- **Callbacks:** Early stopping, learning rate scheduler, and model checkpointing were used to optimize training.

## Model Performance

The model was evaluated on a held-out test set. Below is the key performance metric:

- **Test Accuracy:** 71%

This accuracy reflects the model's ability to correctly identify real and deepfake images.

## Usage

You can use this model for inference by loading the model and running predictions on new images. Below is an example using TensorFlow/Keras:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('cnn_model.h5')

# Load and preprocess the image
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
prediction = model.predict(img_array)
print('Real' if prediction[0][0] < 0.5 else 'Fake')
```

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://huggingface.co/MaanVad3r/DeepFake-Detector
    cd DeepFake-Detection-model.git
    ```

2. **Run Inference**:
    Use the provided script or the sample code above to run inference on your images.

## License

This project is licensed under the MIT License. Feel free to use and modify the model as needed.

---