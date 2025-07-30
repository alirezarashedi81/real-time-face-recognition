import os
import numpy as np
import pickle
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import joblib

embedding_root = "data/embeddings"
X, Y = [], []

# Load all embeddings
for file in os.listdir(embedding_root):
    if not file.endswith(".pkl"): continue

    label = file.replace('.pkl', '')
    with open(os.path.join(embedding_root, file), 'rb') as f:
        embeddings = pickle.load(f)
        X.extend(embeddings)
        Y.extend([label] * len(embeddings))

X = np.array(X)
Y = np.array(Y)

# Normalize
normalizer = Normalizer(norm='l2')
X_norm = normalizer.transform(X)

# Train SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_norm, Y)

# Save models
joblib.dump(svm, "models/SVC_face_recognition.pkl")
joblib.dump(normalizer, "models/normalizer.pkl")

print("✅ SVM model and normalizer saved.")
