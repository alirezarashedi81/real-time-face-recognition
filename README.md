
# 🎯 Real-Time Face Recognition with YOLOv8 + FaceNet + SVM

A powerful, modular, and real-time face recognition system using:
- 🔍 **YOLOv8** for face detection
- 🧠 **Keras-FaceNet** for generating face embeddings
- 🤖 **SVM classifier** for identity recognition

---

## 🚀 Features
- Real-time face detection using your webcam
- Fast and accurate face embedding extraction with FaceNet
- Robust identity classification with SVM
- Modular, clean code structure for easy reuse and extension
- Use precomputed face embeddings—no need to upload the original dataset

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

---

## 📦 Usage

1. **Prepare Embeddings**
   - Place your precomputed embeddings file (for example, `embeddings.npy` or `embeddings.pkl`) in the project directory or the expected location.
   - Make sure your embeddings file matches the format expected by the SVM classifier.

2. **Run the Real-Time Recognition**
   - Use the main script to start recognition using your webcam:
     ```bash
     python main.py --embeddings path/to/your/embeddings.npy
     ```
   - The script will:
     - Open your webcam
     - Detect faces in real-time with YOLOv8
     - Generate embeddings with FaceNet
     - Classify identities using the SVM, based on your supplied embeddings

3. **(Optional) Update Embeddings**
   - If you want to add new faces, generate new embeddings and update your embeddings file according to the project’s format.

---

## ⚙️ Configuration

- You can specify paths and parameters (such as webcam index, threshold, etc.) as command-line arguments or in a configuration file.  
  Check `main.py` for available options.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions or improvements.

---

## 📄 License

This project is licensed under the MIT License.

---

Let me know if you want to add or change anything, or if you’d like the exact commands or code snippets based on your project’s files!
