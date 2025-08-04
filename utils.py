import cv2
import numpy as np
from datetime import datetime

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']


def load_age_model(proto_path: str, model_path: str):
    """Load pre-trained age classification model."""
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)

def detect_faces(frame, face_cascade):
    """Detect faces in a frame using Haar cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def predict_age(face_img, age_net):
    """Predict the age range for a given face image."""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                 (78.4263377603, 87.7689143744, 114.895847746),
                                 swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    return AGE_LIST[age_preds[0].argmax()]


def log_prediction(age, log_file="predictions_log.csv"):
    """Save age prediction with timestamp to a log file."""
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()},{age}\n")
