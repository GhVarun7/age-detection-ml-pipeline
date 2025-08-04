import cv2
from utils import load_age_model, detect_faces, predict_age, log_prediction

# Load model
age_net = load_age_model("model/deploy_age.prototxt", "model/age_net.caffemodel")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame, face_cascade)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()

        # Predict age
        age = predict_age(face_img, age_net)

        # Log the prediction
        log_prediction(age)

        # Draw results
        label = f"Age: {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)

    cv2.imshow("Live Age Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
