[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18729715.svg)](https://doi.org/10.5281/zenodo.18729715)

# Identity-Aware Deepfake Forensic Framework

This repository contains the official implementation of the **Identity-Aware Deepfake Forensic Framework** using 512-dimensional (FaceNet512) biometric embeddings and CNN-based verification.

## 📌 Overview
Unlike standard deepfake detectors that only provide a binary (Real/Fake) classification, this framework introduces **Identity-Awareness**. It uses a "Retrieve-Verify" architecture:
1. **Retrieve Stage**: Extracts facial embeddings and uses FAISS to find identity-consistent matches from a database.
2. **Verify Stage**: A CNN-based model verifies the authenticity of the retrieved candidates.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (recommended for CNN inference)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/](https://github.com/)Sakthi0400J/Face_Morph_Detection.git
   cd [Repo-Name]


###Note:
1. The CNN model is too large for uploading so i gave the other files and license for the model in the DeepFake-Detector folder
