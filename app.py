import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "MobileNetV1": "MobileNetV1_no_dropout.h5",
    "MobileNetV2": "MobileNetV2_no_dropout.h5",
}
CLASS_MAPPING = {0: 'Jahe', 1: 'Kencur', 2: 'Kunyit', 3: 'Lengkuas'}
TARGET_SIZE   = (224, 224)

# Deskripsi singkat tiap rempah
DESCRIPTION = {
    'Jahe': 'Jahe (Zingiber officinale) tidak hanya populer sebagai bumbu, tetapi juga diolah menjadi minuman tradisional penghangat tubuh dengan khasiat meredakan sakit kepala, masuk angin, dan meningkatkan nafsu makan berkat kandungan gingerol, serta kerap diperkaya pewarna alami casiavera (Ayuchecaria et al., 2022).',
    'Kencur': 'Kencur (Kaempferia galanga) memiliki aroma khas, dapat meredakan batuk, flu, sakit kepala, keseleo, radang lambung, hingga memperlancar haid dan mengatasi radang telinga. (Hakim, 2015)',
    'Kunyit': 'Kunyit (Curcuma longa) mengandung kurkumin dan minyak atsiri yang efektif meredakan nyeri gastritis. Senyawa tersebut membantu melapisi dinding lambung yang luka serta menurunkan produksi asam lambung, sehingga bisa mengontrol kelebihan asam di perut (Syafila et al., 2024)',
    'Lengkuas': 'Lengkuas (Alpinia galanga) tidak hanya menambah aroma masakan, tetapi juga kaya akan senyawa bioaktif seperti flavonoid, alkaloid, saponin, dan fenol yang memberikan efek antitumor, antioksidan, antimikroba, penghambatan asam lambung, antiinflamasi, serta mampu menghambat pertumbuhan Klebsiella pneumoniae (Badriyah, Ifandi & Alfiza, 2023).'
}

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

def display_spices_images():
    """Show a 2×2 grid of the four rempah sample images."""
    imgs = load_spice_images()
    cols = st.columns(2)
    for i, name in enumerate(CLASS_MAPPING.values()):
        with cols[i % 2]:
            st.image(imgs[name], caption=name, width=300)

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
    st.markdown(
        """
        <div style="padding: 10px; border-radius: 6px;">
          <p style="margin: 0;">
            Aplikasi ini menggunakan dua varian MobileNet <strong>tanpa dropout</strong>,  
            yaitu V1 dan V2. Keduanya dilengkapi dua lapisan fully-connected,  
            dioptimasi dengan Adam, dan dilatih selama 15 epoch. 
            <strong>MobileNetV1</strong> mencapai akurasi pelatihan sebesar <strong>95%</strong>,  
            sedangkan <strong>MobileNetV2</strong> mencapai <strong>97%</strong>.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("---")
    st.subheader("Gambar Masing-masing Rempah")
    display_spices_images()

# ─── CLASSIFICATION PAGE ────────────────────────────────────────────────────────────
def predict():
    st.title("📸 Classification")
    st.markdown("Unggah gambar rempah untuk prediksi jenisnya.")

    # Pilih model lewat dropdown
    model_name = st.selectbox("Pilih model:", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_name]

    # Load model yang dipilih
    model = load_model(model_path)
    if model is None:
        return

    # Upload dan prediksi
    uploaded = st.file_uploader("Pilih file (jpg/png)", type=["jpg","jpeg","png"])
    if not uploaded:
        st.info("Silakan unggah gambar terlebih dahulu.")
        return

    # preprocess
    img = load_img(uploaded, target_size=TARGET_SIZE)
    arr = img_to_array(img) / 255.0
    st.image(arr, caption="Gambar Masukan", width=300)

    # predict
    probs = model.predict(np.expand_dims(arr, 0))[0]
    idx   = int(np.argmax(probs))
    label = CLASS_MAPPING[idx]
    confidence = probs[idx]

    # Tampilkan hasil dan confidence score
    st.success(
        f"**Prediksi ({model_name}):** {label}\n\n"
        f"**Confidence Score:** {confidence:.2%}\n"
    )
    # Tampilkan deskripsi singkat berdasarkan label
    desc = DESCRIPTION.get(label, "Tidak ada deskripsi tersedia.")
    st.info(desc)

# ─── ABOUT PAGE ─────────────────────────────────────────────────────────────────
def about():
    st.title("❗ Tentang Model")
    st.write("---")
    st.markdown(
        """
        <div style="background-color: #FFF3CD; padding: 10px; border-radius: 5px;">
          <p style="color: #856404; margin: 0;">
            Aplikasi ini menggunakan dua varian MobileNet <strong>tanpa dropout</strong>,  
            yaitu V1 dan V2. Keduanya dilengkapi dua lapisan fully-connected,  
            dioptimasi dengan Adam, dan dilatih selama 15 epoch. 
            MobileNetV1 mencapai akurasi pelatihan sebesar 95%,  
            sedangkan MobileNetV2 mencapai 97%.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("---")
    st.subheader("**Arsitektur Model**")
    col1, col2 = st.columns(2)
    with col1:
        st.image("arc/arch_mobilenet_v1.png", caption="MobileNetV1", width=400)
    with col2:
        st.image("arc/arch_mobilenet_v2.png", caption="MobileNetV2", width=400)

    st.markdown("---")
    st.subheader("**Grafik Pelatihan**")
    st.image("graph/graph_mobilenetv1.png", caption="Training Plot MobileNetV1", use_column_width=True)
    st.image("graph/graph_mobilenetv2.png", caption="Training Plot MobileNetV2", use_column_width=True)

# ─── MAIN ────────────────────────────────────────────────────────────────────────
if menu == "Home":
    home()
elif menu == "Classification":
    predict()
elif menu == "About":
    about()

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.image("remove.png", width=150)
st.sidebar.markdown("© 2025 Najmah Femalea. All rights reserved.", unsafe_allow_html=True)