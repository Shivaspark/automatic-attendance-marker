# Automatic Attendance Marker (Project #10)

The tenth project in my **AI/ML Learning Path**. This project is a comprehensive facial recognition system designed to automate attendance tracking by identifying individuals in a live video stream.

## 📌 Overview
This system integrates face detection, facial feature extraction (embeddings), and high-accuracy classification to create a seamless attendance pipeline. It utilizes pre-trained neural networks for feature mapping and traditional ML for the final classification.

## 🛠️ Tech Stack
* **Language:** Python
* **Detection:** Haar Cascade (Frontal Face)
* **Feature Extraction:** OpenFace (nn4.small2 model)
* **Classifier:** Support Vector Machine (SVM)
* **Serialization:** Pickle (for storing facial embeddings)
* **Libraries:** OpenCV, Scikit-Learn, NumPy

## ⚙️ How It Works
1. **Face Detection:** The system uses Haar Cascade classifiers to detect and crop faces from the webcam feed.
2. **Embedding Generation:** The cropped face is passed through the **OpenFace nn4.small2** model, which generates a 128-dimensional vector (embedding) representing the unique features of the face.
3. **Data Serialization:** During training, these embeddings are stored in **Pickle (.pkl)** files, allowing the system to load known faces instantly during inference.
4. **Classification:** An **SVM classifier** compares the live 128-d embedding against the stored data to predict the identity of the person.
5. **Attendance Logging:** Once a face is recognized with high confidence, the name and timestamp are logged (to a CSV or Database).

## 📂 Project Structure
- `extract_embeddings.py`: Generates the 128-d vectors and saves them via Pickle.
- `train_model.py`: Trains the SVM classifier on the extracted embeddings.
- `recognize_video.py`: The main script for real-time detection and attendance marking.

## 🚀 Quick Start
1. **Install Dependencies:**
   ```bash
   pip install opencv-python scikit-learn numpy
2. **Prepare the dataset**
   Add images of individuals into the dataset/ folder.
3. **Run Pipeline**
   ```bash
   python 2_preprocessingEmbeddings.py
   python 3_trainingFaceML.py
   Python 5_recognizingPersonwithCSVData

*Progress: Successfully integrated deep learning embeddings with classical machine learning classifiers.*
