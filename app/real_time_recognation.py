import cv2
import numpy as np
from keras_facenet import FaceNet
from ultralytics import YOLO
import joblib

# Load models
svm = joblib.load("models/SVC_face_recognition.pkl")
scaler = joblib.load("models/normalizer.pkl")
detector = YOLO("models/yolov8n-face.pt")
embedder = FaceNet()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = detector(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]

        try:
            face_resized = cv2.resize(face, (160, 160))
            emb = embedder.embeddings([face_resized])[0]
            emb_norm = scaler.transform([emb])
            probs = svm.predict_proba(emb_norm)[0]
            label_idx = np.argmax(probs)
            confidence = probs[label_idx]
            name = svm.classes_[label_idx] if confidence > 0.75 else "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        except:
            continue

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
