# =========================================
# Streamlit App - Klasifikasi Bunga Vany 🌸
# =========================================

import streamlit as st
import cv2
import numpy as np
import mahotas
from skimage.feature import hog
import pickle
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1. Fungsi Resize + Padding
# ===============================
def resize_with_padding(image, target_size=256):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - top
    left, right = delta_w // 2, delta_w - left
    result = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return result

# ===============================
# 2. Normalisasi
# ===============================
def normalize_image(img):
    return img.astype('float32') / 255.0

# ===============================
# 3. Ekstraksi Fitur
# ===============================
def extract_hsv_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(16,16),
                      cells_per_block=(2,2), block_norm='L2-Hys', visualize=True)
    return features

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray, return_mean=True)

# ===============================
# 4. Load Model
# ===============================
MODEL_PATH = "model_lightgbm.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"⚠️ File model tidak ditemukan: `{MODEL_PATH}`. Pastikan file ada di GitHub.")
    st.stop()

# ===============================
# 5. Label Kelas
# ===============================
classes = ["Bunga Mawar", "Bunga Tulip", "Bunga Dandelion", "Bunga Daisy", "Bunga Matahari"]

# ===============================
# 6. Streamlit UI
# ===============================
st.title("🌸 Prediksi Klasifikasi Bunga")
st.write("Upload gambar bunga, dan model akan memprediksi jenisnya.")

uploaded_file = st.file_uploader("📁 Upload gambar (.jpg/.png)", type=["jpg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar diunggah", use_column_width=True)

    # Konversi → OpenCV
    img = np.array(image)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Preprocessing & Ekstraksi fitur
    img_resized = resize_with_padding(img, 256)
    img_norm = normalize_image(img_resized)
    img_uint8 = (img_norm * 255).astype("uint8")

    hsv_feat = extract_hsv_features(img_norm)
    hog_feat = extract_hog_features(img_norm)
    glcm_feat = extract_glcm_features(img_uint8)

    features = np.concatenate([hsv_feat, hog_feat, glcm_feat]).reshape(1, -1)

    # Prediksi
    prediction = model.predict(features)[0]
    st.success(f"🌼 Hasil Prediksi: **{classes[prediction]}**")
