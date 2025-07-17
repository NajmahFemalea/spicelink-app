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

def display_spices():
    """Tampilkan grid 2×2 gambar rempah beserta deskripsinya."""
    # Muat semua gambar ke dalam dict {nama: path}
    spice_dir = "datatest"
    imgs = {
        'Kunyit':  os.path.join(spice_dir, "kunyit.jpeg"),
        'Kencur':  os.path.join(spice_dir, "kencur.jpeg"),
        'Jahe':    os.path.join(spice_dir, "jahe.jpeg"),
        'Lengkuas':os.path.join(spice_dir, "lengkuas.jpeg"),
    }

    # Deskripsi untuk tiap rempah
    descriptions = {
        'Kunyit': 'Kunyit (_Curcuma longa_) mengandung kurkumin dan minyak atsiri yang efektif meredakan nyeri gastritis. Senyawa tersebut membantu melapisi dinding lambung yang luka serta menurunkan produksi asam lambung, sehingga bisa mengontrol kelebihan asam di perut (Syafila et al., 2024)',
        'Kencur': 'Kencur (_Kaempferia galanga_) Selain memperkaya cita rasa, kencur juga diolah menjadi jamu tradisional seperti beras kencur karena khasiatnya, meredakan batuk, flu, sakit kepala, keseleo, radang lambung, hingga memperlancar haid dan mengatasi radang telinga. (Hakim, 2015)',
        'Jahe': 'Jahe (_Zingiber officinale_) tidak hanya populer sebagai bumbu, tetapi juga diolah menjadi minuman tradisional penghangat tubuh dengan khasiat meredakan sakit kepala, masuk angin, dan meningkatkan nafsu makan berkat kandungan gingerol, serta kerap diperkaya pewarna alami casiavera (Ayuchecaria et al., 2022).',
        'Lengkuas': 'Lengkuas (_Alpinia galanga_) tidak hanya menambah aroma masakan, tetapi juga kaya akan senyawa bioaktif seperti flavonoid, alkaloid, saponin, dan fenol yang memberikan efek antitumor, antioksidan, antimikroba, penghambatan asam lambung, antiinflamasi, serta mampu menghambat pertumbuhan Klebsiella pneumoniae (Badriyah, Ifandi & Alfiza, 2023).'
    }

    cols = st.columns(2)
    for i, (name, img_path) in enumerate(imgs.items()):
        with cols[i % 2]:
            st.image(img_path, caption=name, use_container_width=True)
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
    st.markdown(
        """
        <div style="padding: 10px; border-radius: 6px;">
          <p style="margin: 0;">
            <strong>SpiceLinK</strong> adalah aplikasi web interaktif yang dirancang untuk memudahkan klasifikasi empat jenis rimpang populer di 
            Indonesia--Jahe, Kunyit, Kencur, dan Lengkuas-- (CNNIndonesia, 2024) hanya dengan mengunggah satu gambar🌿📸. 
            Selain klasifikasi otomatis, antarmuka aplikasi juga menampilkan gambar rempah serta manfaatnya, statistik akurasi pelatihan, dan visualisasi arsitektur, 
            sehingga pengguna dapat memahami proses kerja model secara transparan. Dengan navigasi yang intuitif dan hasil klasifikasi real‑time, SpiceLinK siap membantu kamu. 
            Yuk, coba SpiceLinK!✨🚀
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("---")
    st.subheader("Gambar Masing-masing Rempah")
    display_spices()

# ─── CLASSIFICATION PAGE ────────────────────────────────────────────────────────────
def classify():
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
        f"**Classification ({model_name}):** {label}\n\n"
        f"**Confidence Score:** {confidence:.2%}\n"
    )

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
    st.image("graph/graph_mobilenetv1.png", caption="Training Plot MobileNetV1", use_container_width=True)
    st.image("graph/graph_mobilenetv2.png", caption="Training Plot MobileNetV2", use_container_width=True)

# ─── MAIN ────────────────────────────────────────────────────────────────────────
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
