import os
import cv2
from ultralytics import YOLO
from keras_facenet import FaceNet
import pickle

# Load models
detector = YOLO("models/yolov8n-face.pt")
embedder = FaceNet()

# Define paths
input_root = "data/raw"
output_root = "data/cropped"
embedding_root = "data/embeddings"

os.makedirs(output_root, exist_ok=True)
os.makedirs(embedding_root, exist_ok=True)

# Process each person folder
for person in os.listdir(input_root):
    person_folder = os.path.join(input_root, person)
    if not os.path.isdir(person_folder): continue

    output_folder = os.path.join(output_root, person)
    os.makedirs(output_folder, exist_ok=True)

    embeddings = []

    for img_name in os.listdir(person_folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue

        img_path = os.path.join(person_folder, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        results = detector(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            print(f"[!] No face in {img_name}")
            continue

        x1, y1, x2, y2 = map(int, boxes[0])
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))

        cv2.imwrite(os.path.join(output_folder, img_name), face)

        embedding = embedder.embeddings([face])[0]
        embeddings.append(embedding)

    with open(os.path.join(embedding_root, f'{person}.pkl'), "wb") as f:
        pickle.dump(embeddings, f)

print("✅ Cropping and embedding done.")
