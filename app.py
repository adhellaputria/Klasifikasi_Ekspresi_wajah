import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

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
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
        max-width: 700px;
    }
    .header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .result-senang {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .result-sedih {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 2px solid #17a2b8;
    }
    .result-marah {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Judul & Deskripsi
# ─────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center;'>😊 Klasifikasi Ekspresi Wajah</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Gunakan AI untuk mengenali ekspresi wajah Anda: Senang, Sedih, atau Marah</p>", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model CNN yang sudah dilatih"""
    try:
        # Coba load dari model yang sudah tersimpan
        model_path = 'models/cnn_model.h5'
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            st.warning("⚠️ Model belum tersedia. Gunakan notebook untuk melatih model terlebih dahulu.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ─────────────────────────────────────────────────
# Load Kelas
# ─────────────────────────────────────────────────
CLASS_NAMES = ['MARAH', 'SEDIH', 'SENANG']
EMOJIS = {
    'MARAH': '😠',
    'SEDIH': '😢',
    'SENANG': '😊'
}
COLORS = {
    'MARAH': 'result-marah',
    'SEDIH': 'result-sedih',
    'SENANG': 'result-senang'
}

# ─────────────────────────────────────────────────
# Fungsi Preprocessing
# ─────────────────────────────────────────────────
def preprocess_image(image, img_size=224):
    """Preprocessing gambar untuk model CNN"""
    img = image.resize((img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    return img_array

# ─────────────────────────────────────────────────
# Sidebar - Upload Options
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
        type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'],
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
        
        # Load model
        model = load_model()
        
        if model is not None:
            # Tombol Prediksi
            if st.button("🔍 Deteksi Ekspresi", use_container_width=True, type="primary"):
                with st.spinner("⏳ Menganalisis ekspresi Anda..."):
                    try:
                        # Preprocessing
                        img_array = preprocess_image(image)
                        
                        # Prediksi
                        prediction = model.predict(img_array, verbose=0)
                        predicted_idx = np.argmax(prediction[0])
                        predicted_class = CLASS_NAMES[predicted_idx]
                        confidence = prediction[0][predicted_idx] * 100
                        
                        # Tampilkan hasil
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
                        
                        # Detail probabilitas semua kelas
                        st.markdown("#### 📊 Detail Probabilitas:")
                        prob_data = []
                        for i, class_name in enumerate(CLASS_NAMES):
                            prob = prediction[0][i] * 100
                            prob_data.append({
                                "Ekspresi": class_name,
                                "Probabilitas": f"{prob:.2f}%"
                            })
                        
                        col1, col2, col3 = st.columns(3)
                        for i, (class_name, prob) in enumerate([(CLASS_NAMES[j], prediction[0][j]*100) for j in range(len(CLASS_NAMES))]):
                            with [col1, col2, col3][i]:
                                st.metric(class_name, f"{prob:.1f}%")
                        
                        # Grafik probabilitas
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors_bar = ['#dc3545', '#17a2b8', '#28a745']
                        bars = ax.bar(CLASS_NAMES, prediction[0]*100, color=colors_bar, alpha=0.7)
                        ax.set_ylabel('Probabilitas (%)', fontweight='bold')
                        ax.set_title('Distribusi Probabilitas Prediksi', fontweight='bold')
                        ax.set_ylim(0, 100)
                        
                        for bar, prob in zip(bars, prediction[0]*100):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                   f'{prob:.1f}%', ha='center', fontweight='bold')
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Error saat prediksi: {e}")
        else:
            st.warning(
                "⚠️ Model tidak tersedia.\n\n"
                "Langkah yang diperlukan:\n"
                "1. Jalankan notebook `Klasifikasi_Ekspresi_Wajah_Colab.ipynb`\n"
                "2. Latih model CNN atau Transfer Learning\n"
                "3. Simpan model ke folder `models/`\n"
                "4. Deploy app ini ke Streamlit"
            )

# ─────────────────────────────────────────────────
# Ambil dari Kamera
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
                        img_array = preprocess_image(image)
                        prediction = model.predict(img_array, verbose=0)
                        predicted_idx = np.argmax(prediction[0])
                        predicted_class = CLASS_NAMES[predicted_idx]
                        confidence = prediction[0][predicted_idx] * 100
                        
                        st.success("✅ Analisis selesai!")
                        
                        result_emoji = EMOJIS[predicted_class]
                        result_color = COLORS[predicted_class]
                        st.markdown(f"""
                        <div class='result-box {result_color}'>
                            <p style='font-size: 36px; margin: 10px 0;'>{result_emoji}</p>
                            <p style='font-size: 24px; margin: 10px 0;'>{predicted_class}</p>
                            <p style='font-size: 18px; margin: 10px 0;'>Kepercayaan: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# ─────────────────────────────────────────────────
# Informasi & Footer
# ─────────────────────────────────────────────────
st.divider()
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Informasi
**Model:** CNN From Scratch atau Transfer Learning (MobileNetV2)

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
