import os
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "MobileNetV1": "MobileNetV1_no_dropout.h5",
    "MobileNetV2": "MobileNetV2_no_dropout.h5",
}
CLASS_MAPPING = {0: 'Jahe', 1: 'Kencur', 2: 'Kunyit', 3: 'Lengkuas'}
TARGET_SIZE   = (224, 224)

# ─── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_spice_images():
    """Cache paths to sample spice images."""
    spice_dir = "datatest"
    return {
        name: os.path.join(spice_dir, f"{name.lower()}.jpeg")
        for name in CLASS_MAPPING.values()
    }

@st.cache_resource(show_spinner=False, max_entries=4)
def load_model(path):
    """Load a .h5 model given its path (cached per path)."""
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Error loading model:\n{e}")
        return None

def compress_image(file, max_size_kb=200):
    """
    Kompres file upload (BytesIO) ke JPEG dengan kualitas menurun
    hingga ukuran di bawah max_size_kb KB.
    Kembalikan BytesIO baru.
    """
    img = Image.open(file)
    buf = io.BytesIO()
    quality = 85

    # Turunkan kualitas secara bertahap hingga target tercapai
    while True:
        buf.seek(0)
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        size_kb = buf.tell() / 1024
        if size_kb <= max_size_kb or quality < 30:
            break
        quality -= 5

    buf.seek(0)
    return buf

def display_spices():
    """Tampilkan grid 2×2 gambar rempah beserta deskripsinya."""
    spice_dir = "datatest"
    imgs = {
        'Kunyit':  os.path.join(spice_dir, "kunyit.jpeg"),
        'Kencur':  os.path.join(spice_dir, "kencur.jpeg"),
        'Jahe':    os.path.join(spice_dir, "jahe.jpeg"),
        'Lengkuas':os.path.join(spice_dir, "lengkuas.jpeg"),
    }
    descriptions = {
        'Kunyit': 'Kunyit (_Curcuma longa_) …',
        'Kencur': 'Kencur (_Kaempferia galanga_) …',
        'Jahe':   'Jahe (_Zingiber officinale_) …',
        'Lengkuas': 'Lengkuas (_Alpinia galanga_) …'
    }

    cols = st.columns(2)
    for i, (name, img_path) in enumerate(imgs.items()):
        with cols[i % 2]:
            st.image(img_path, caption=name, use_column_width=True)
            st.markdown(f"**{name}**  \n{descriptions[name]}")

# ─── Layout ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔍 SpiceLinK",
    layout="wide",
    initial_sidebar_state="expanded"
)

menu = st.sidebar.selectbox("📂 Menu", ["Home", "Classification", "About"])

# ─── HOME PAGE ─────────────────────────────────────────────────────────────────
def home():
    st.title("Selamat Datang di SpiceLinK ✨")
    st.write("---")
    st.markdown("""
      <div style="padding:10px; border-radius:6px;">
        <strong>SpiceLinK</strong> …
      </div>
    """, unsafe_allow_html=True)
    st.write("---")
    st.subheader("Gambar Masing-masing Rempah")
    display_spices()

# ─── CLASSIFICATION PAGE ─────────────────────────────────────────────────────────
def classify():
    st.title("📸 Classification")
    st.markdown("Unggah gambar rempah untuk prediksi jenisnya.")

    model_name = st.selectbox("Pilih model:", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_name]
    model = load_model(model_path)
    if model is None:
        return

    uploaded = st.file_uploader("Pilih file (jpg/png)", type=["jpg","jpeg","png"])
    if not uploaded:
        st.info("Silakan unggah gambar terlebih dahulu.")
        return

    # Kompres dulu agar file lebih kecil 
    compressed = compress_image(uploaded, max_size_kb=200)

    # Preprocess
    img = load_img(compressed, target_size=TARGET_SIZE)
    arr = img_to_array(img) / 255.0
    st.image(arr, caption="Gambar Masukan", width=300)

    # Predict
    probs = model.predict(np.expand_dims(arr, 0))[0]
    idx = int(np.argmax(probs))
    label = CLASS_MAPPING[idx]
    confidence = probs[idx]

    st.success(f"**Classification ({model_name}):** {label}\n\n**Confidence Score:** {confidence:.2%}")

# ─── ABOUT PAGE ─────────────────────────────────────────────────────────────────
def about():
    st.title("❗ Tentang Model")
    st.write("---")
    st.markdown("""
      <div style="background-color:#FFF3CD;padding:10px;border-radius:5px;">
        Aplikasi ini menggunakan dua varian MobileNet tanpa dropout…
      </div>
    """, unsafe_allow_html=True)
    st.write("---")
    st.subheader("**Arsitektur Model**")
    col1, col2 = st.columns(2)
    with col1:
        st.image("arc/arch_mobilenet_v1.png", caption="MobileNetV1", width=400)
    with col2:
        st.image("arc/arch_mobilenet_v2.png", caption="MobileNetV2", width=400)
    st.markdown("---")
    st.subheader("**Grafik Pelatihan**")
    st.image("graph/graph_mobilenetv1.png", use_column_width=True)
    st.image("graph/graph_mobilenetv2.png", use_column_width=True)

# ─── MAIN ───────────────────────────────────────────────────────────────────────
if menu == "Home":
    home()
elif menu == "Classification":
    classify()
elif menu == "About":
    about()

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.image("remove.png", width=150)
st.sidebar.markdown("© 2025 Najmah Femalea. All rights reserved.", unsafe_allow_html=True)
