import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ─────────────────────────────────────────────────
# Konfigurasi Streamlit
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="😊 Klasifikasi Ekspresi Wajah",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 700px; }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .result-senang { background-color: #d4edda; color: #155724; border: 2px solid #28a745; }
    .result-sedih  { background-color: #d1ecf1; color: #0c5460; border: 2px solid #17a2b8; }
    .result-marah  { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Judul & Deskripsi
# ─────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center;'>😊 Klasifikasi Ekspresi Wajah</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Gunakan AI untuk mengenali ekspresi wajah Anda: Senang, Sedih, atau Marah</p>", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────
CLASS_NAMES = ['MARAH', 'SEDIH', 'SENANG']
EMOJIS  = {'MARAH': '😠', 'SEDIH': '😢', 'SENANG': '😊'}
COLORS  = {'MARAH': 'result-marah', 'SEDIH': 'result-sedih', 'SENANG': 'result-senang'}

# ─── GANTI dengan File ID model kamu dari Google Drive ───
GDRIVE_FILE_ID = "19KXN4-4auyo9qXzmg-VhBLqWqzuBDuFp"
# ─────────────────────────────────────────────────────────

MODEL_PATH = "models/cnn_model.h5"

# ─────────────────────────────────────────────────
# Load Model (dengan auto-download dari Google Drive)
# ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        if GDRIVE_FILE_ID == "FILE_ID_KAMU":
            st.error("❌ FILE_ID Google Drive belum diisi di app.py.")
            return None

        try:
            with st.spinner("⏳ Mengunduh model dari Google Drive... (hanya sekali)"):
                url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"❌ Gagal mengunduh model: {e}")
            return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

# ─────────────────────────────────────────────────
# Fungsi Preprocessing
# ─────────────────────────────────────────────────
def preprocess_image(image, img_size=224):
    img = image.convert("RGB").resize((img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    return img_array

# ─────────────────────────────────────────────────
# Fungsi Tampilkan Hasil
# ─────────────────────────────────────────────────
def show_result(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = prediction[0][predicted_idx] * 100

    st.success("✅ Analisis selesai!")

    # Hasil utama
    result_emoji = EMOJIS[predicted_class]
    result_color = COLORS[predicted_class]
    st.markdown(f"""
    <div class='result-box {result_color}'>
        <p style='font-size: 36px; margin: 10px 0;'>{result_emoji}</p>
        <p style='font-size: 24px; margin: 10px 0;'>{predicted_class}</p>
        <p style='font-size: 18px; margin: 10px 0;'>Kepercayaan: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Detail Probabilitas:")
    col1, col2, col3 = st.columns(3)
    for i, col in enumerate([col1, col2, col3]):
        with col:
            st.metric(CLASS_NAMES[i], f"{prediction[0][i]*100:.1f}%")

    # Grafik
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_NAMES, prediction[0] * 100,
                  color=['#dc3545', '#17a2b8', '#28a745'], alpha=0.75)
    ax.set_ylabel('Probabilitas (%)', fontweight='bold')
    ax.set_title('Distribusi Probabilitas Prediksi', fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, prob in zip(bars, prediction[0] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{prob:.1f}%', ha='center', fontweight='bold')
    st.pyplot(fig)

# ─────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────
st.sidebar.header("📤 Upload Foto")
st.sidebar.markdown("Pilih metode upload:")
upload_method = st.sidebar.radio(
    "Pilih metode:",
    ["📸 Upload File", "📷 Ambil dari Kamera"],
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────────
# Upload File
# ─────────────────────────────────────────────────
if upload_method == "📸 Upload File":
    uploaded_file = st.file_uploader(
        "Pilih foto ekspresi wajah",
        type=['jpg', 'jpeg', 'png'],
        help="Format: JPG, JPEG, atau PNG"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="📸 Foto Anda", use_column_width=True)
        with col2:
            st.metric("Ukuran File", f"{uploaded_file.size / 1024:.1f} KB")
            st.metric("Format", uploaded_file.type.split('/')[-1].upper())

        st.divider()
        model = load_model()

        if model is not None:
            if st.button("🔍 Deteksi Ekspresi", use_container_width=True, type="primary"):
                with st.spinner("⏳ Menganalisis ekspresi Anda..."):
                    try:
                        show_result(image, model)
                    except Exception as e:
                        st.error(f"❌ Error saat prediksi: {e}")

# ─────────────────────────────────────────────────
# Kamera
# ─────────────────────────────────────────────────
else:
    picture = st.camera_input("Ambil foto menggunakan kamera")

    if picture is not None:
        image = Image.open(picture)
        st.image(image, caption="📷 Foto dari Kamera", use_column_width=True)
        st.divider()

        model = load_model()

        if model is not None:
            if st.button("🔍 Deteksi Ekspresi", use_container_width=True, type="primary"):
                with st.spinner("⏳ Menganalisis ekspresi Anda..."):
                    try:
                        show_result(image, model)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# ─────────────────────────────────────────────────
# Sidebar Info
# ─────────────────────────────────────────────────
st.divider()
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Informasi
**Model:** CNN / Transfer Learning (MobileNetV2)
**Kelas:**
- 😊 SENANG
- 😢 SEDIH
- 😠 MARAH

**Dataset:** ~300 gambar ekspresi wajah

**Teknologi:**
- TensorFlow & Keras
- Streamlit
- Python 3.9+
""")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dibuat oleh:** Adhella Putria  
**GitHub:** [adhellaputria](https://github.com/adhellaputria)  
**Dibuat:** 2026
""")
