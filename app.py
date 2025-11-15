import os
import requests
import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?export=download&id=13DYqymdg2vASLT96ee0HN_zCsqvAbCTN"
MODEL_PATH = "model.h5"


def download_model():
    if os.path.exists(MODEL_PATH):
        print("✔ Model already exists — skipping download.")
        return

    print("⬇ Downloading model from Google Drive...")
    response = requests.get(MODEL_URL, stream=True)

    if response.status_code != 200:
        raise Exception("❌ Failed to download model")

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print("✔ Model downloaded successfully!")


print("🚀 Starting server...")
download_model()

print("📦 Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✔ Model loaded!")


@app.route("/")
def home():
    return "Model API is running! 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    # الإنجليزية
    result_label = "Aggressive" if prediction > 0.5 else "Not Aggressive"

    # العربية
    result_label_ar = "عدواني" if prediction > 0.5 else "غير عدواني"

    return jsonify({
        "aggressive_probability": float(prediction),
        "label_en": result_label,
        "label_ar": result_label_ar
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
