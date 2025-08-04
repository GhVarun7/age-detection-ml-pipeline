import os
import cv2
import csv
from utils import load_age_model, predict_age

# Load model
age_net = load_age_model("model/deploy_age.prototxt", "model/age_net.caffemodel")

# Define path
input_folder = "images"
output_file = "batch_predictions.csv"

# Create/Open CSV log file
with open(output_file, mode="w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "predicted_age"])

    # Loop through each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(input_folder, filename)
            image = cv2.imread(path)

            if image is None:
                print(f"❌ Failed to load: {filename}")
                continue

            try:
                # Resize and predict
                face_img = cv2.resize(image, (227, 227))
                age = predict_age(face_img, age_net)

                writer.writerow([filename, age])
                print(f"✅ {filename}: {age}")
            except Exception as e:
                print(f"⚠️ Error on {filename}: {e}")
