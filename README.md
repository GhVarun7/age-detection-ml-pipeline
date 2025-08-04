# 🚀 Age Detection ML Pipeline – Tesla-Inspired Diagnostic System

This project simulates Tesla’s NVH (Noise Vibration Harshness) diagnostic workflow using a modular age detection system. Built with OpenCV and PyTorch, it supports real-time webcam detection, offline batch processing, CNN training, and model evaluation.

## 🧠 Features

✅ Real-time age detection with OpenCV  
✅ Batch prediction pipeline using Caffe model  
✅ PyTorch Dataset loader and CNN model training  
✅ Evaluation script for testing trained models  
✅ CSV logging for diagnostics and analysis  
✅ Modular, scalable architecture for production-style ML

---

## 🗂️ Project Structure

```bash
age-detection/
├── images/                # Folder for input face images
├── model/                 # Pretrained Caffe files
├── utils.py               # Modular ML helper functions
├── main.py                # Live age detection via webcam
├── batch_predict.py       # Batch predictions from image folder
├── train.py               # Train CNN using labeled image dataset
├── evaluate.py            # Evaluate CNN on test images
├── torch_dataset.py       # Custom PyTorch dataset class
├── cnn_model.py           # Simple CNN for age group classification
├── labels.csv             # Labeled image data (age group 0–7)
├── predictions_log.csv    # Live webcam log (auto-generated)
├── batch_predictions.csv  # Batch prediction log (auto-generated)
