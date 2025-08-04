# 🎯 Tesla-Inspired ML Pipeline: Real-Time & Offline Age Detection

## 🎯 Problem Statement

Real-time diagnostics in automotive systems (like Tesla’s NVH analysis) require scalable ML pipelines that can detect, classify, and log critical patterns (e.g., noise, events, or behavior). Traditional models often fail to handle both real-time and batch processing, limiting scalability.

## 💡 Solution

This project simulates a production-grade diagnostic system for **age classification** using:
- Real-time webcam input
- Offline batch inference
- Custom CNN training with PyTorch
- Logging and evaluation pipeline

It mimics how a Tesla engineer would build, deploy, and scale perception-based ML models.

---

## 🔑 Key Features

🧠 Real-time webcam detection  
📦 Batch inference with Caffe model  
🧱 Trainable CNN with PyTorch  
📊 Prediction logging for diagnostics  
🚀 Fully modular ML pipeline  
📁 Scalable design for data and models

---

## 🏗️ Architecture

### 1. Real-Time Inference
- Uses OpenCV for webcam feed
- Caffe-based age detection model
- Overlays age label on live video
- Logs predictions to CSV

### 2. Batch Processing
- Loads a folder of images
- Predicts age in bulk using the same Caffe model
- Logs results to `batch_predictions.csv`

### 3. CNN Training & Evaluation
- PyTorch Dataset (`AgeLabeledDataset`)
- Custom CNN (`cnn_model.py`) with 2 conv layers
- Model training from `labels.csv`
- Evaluation on new unseen images

---

## 🛠️ Tech Stack

- Python
- OpenCV (live webcam + image preprocessing)
- PyTorch (CNN training + inference)
- Pandas (data handling & logging)
- NumPy (array ops)
- Caffe (pre-trained model)
- Torchvision (dataset transforms)

---

## 📊 Performance Metrics (Example)

| Metric              | Result           |
|---------------------|------------------|
| Webcam inference    | ~30 FPS          |
| Batch inference     | <0.2 sec/image   |
| CNN Accuracy        | 87% (on sample)  |
| Logging             | CSV-based        |
| Scalability         | Easily extendable for 10k+ images |

---

## 📈 Results (Simulated)
<img width="505" height="248" alt="image" src="https://github.com/user-attachments/assets/29ac6ebb-5c40-43ba-bd47-0885a4dcee65" />


🔄 Future Improvements
-Add audio event detection (for Tesla NVH alignment)
-Export CNN model to ONNX / TorchScript
-Add Streamlit or Flask GUI
-Real-time face tracking + age + emotion detection
-Deploy to Jetson Nano / Raspberry Pi for edge inference



