import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load model and scaler
with open("best_classifier.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

THRESHOLD = 0.45430534517214627

def user_input_features():
    st.sidebar.header('Input Manual')

    cp = st.sidebar.selectbox('Jenis Nyeri Dada', options=[1, 2, 3, 4],
                              format_func=lambda x: ("Angina" if x == 1 else 
                                                     "Angina Atipikal" if x == 2 else
                                                     "Nyeri Non-anginal" if x == 3 else
                                                     "Asimtomatik"))

    thalach = st.sidebar.slider("Detak Jantung Maksimum yang Dicapai", 71, 202, 150)
    slope = st.sidebar.selectbox("Kemiringan Segmen ST Puncak Latihan", options=[0, 1, 2],
                                 format_func=lambda x: ("Meningkat" if x == 0 else 
                                                        "Datar" if x == 1 else
                                                        "Menurun"))

    oldpeak = st.sidebar.slider("Depresi ST yang Diinduksi oleh Latihan", 0.0, 6.2, 1.0)
    exang = st.sidebar.selectbox("Angina yang Diinduksi oleh Latihan", options=[0, 1],
                                 format_func=lambda x: "Tidak" if x == 0 else "Ya")

    ca = st.sidebar.slider("Jumlah Pembuluh Darah Utama", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3],
                                format_func=lambda x: ("Normal" if x == 1 else 
                                                       "Defek Tetap" if x == 2 else
                                                       "Defek Reversibel"))

    sex = st.sidebar.selectbox("Jenis Kelamin", options=['Perempuan', 'Pria'])
    sex = 0 if sex == 'Perempuan' else 1

    age = st.sidebar.slider("Usia", 29, 77, 55)

    data = {'ca': ca, 'oldpeak': oldpeak, 'exang': exang, 'thal': thal,
            'sex': sex, 'age': age, 'slope': slope, 'cp': cp, 'thalach': thalach}

    features = pd.DataFrame(data, index=[0])
    return features

def heart():
    st.write("""
    ### Vital Guard 🛡️ : Prediksi Penyakit Jantung
    
    Aplikasi ini memprediksi **Penyakit Jantung** berdasarkan data pengguna.
    
    Data diperoleh dari [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) oleh UCIML.
    """)

st.set_page_config(page_title="Vital Guard", page_icon="🛡️")

# Header and image
st.write("""
### Vital Guard 🛡️ : 
### Prediksi Penyakit Jantung

Aplikasi ini memprediksi **Penyakit Jantung** berdasarkan data pengguna.

Data diperoleh dari [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) oleh UCIML.
""")

st.image('download.jpeg', width=250)

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #2e3b4e;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        width: 100%;
        border-radius: 12px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button:focus {
        background-color: #388E3C;
    }
    .stImage img {
        margin: 0 auto;
    }
    .stNumberInput input {
        background-color: #2e3b4e;
        color: white;
    }
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    .footer {
        text-align: center;
        padding: 20px;
        position: fixed;
        bottom: 0;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.expander("Deskripsi Fitur", expanded=False):
    st.markdown("""
    1. **Jenis Nyeri Dada (<span style='color: #4CAF50;'>cp</span>)**:
       - Angina: Nyeri dada yang disebabkan oleh kekurangan aliran darah ke jantung.
       - Angina Atipikal: Nyeri dada yang tidak sesuai dengan pola khas angina.
       - Nyeri Non-anginal: Nyeri dada yang bukan disebabkan oleh penyakit jantung.
       - Asimtomatik: Tidak ada nyeri dada yang dirasakan.

    2. **Detak Jantung Maksimum yang Dicapai (<span style='color: #4CAF50;'>thalach</span>)**:
       - Jumlah maksimum detak jantung yang dicapai saat berolahraga.

    3. **Kemiringan Segmen ST Puncak Latihan (<span style='color: #4CAF50;'>slope</span>)**:
       - Meningkat: Kemiringan segmen ST meningkat setelah latihan.
       - Datar: Kemiringan segmen ST tetap datar setelah latihan.
       - Menurun: Kemiringan segmen ST menurun setelah latihan.

    4. **Depresi ST yang Diinduksi oleh Latihan (<span style='color: #4CAF50;'>oldpeak</span>)**:
       - Penurunan segmen ST pada EKG selama latihan, menunjukkan masalah dengan aliran darah.

    5. **Angina yang Diinduksi oleh Latihan (<span style='color: #4CAF50;'>exang</span>)**:
       - Tidak: Tidak mengalami angina selama latihan.
       - Ya: Mengalami angina selama latihan.

    6. **Jumlah Pembuluh Darah Utama (<span style='color: #4CAF50;'>ca</span>)**:
       - Jumlah pembuluh darah utama yang terlihat dalam sinar-X dengan kontras.

    7. **Thalassemia (<span style='color: #4CAF50;'>thal</span>)**:
       - Normal: Hasil tes thalassemia normal.
       - Defek Tetap: Defek thalassemia tetap.
       - Defek Reversibel: Defek thalassemia yang dapat kembali normal.

    8. **Jenis Kelamin (<span style='color: #4CAF50;'>sex</span>)**:
       - Perempuan: Jenis kelamin perempuan.
       - Pria: Jenis kelamin pria.

    9. **Usia (<span style='color: #4CAF50;'>age</span>)**:
       - Usia pasien dalam tahun.
    """, unsafe_allow_html=True)

df = user_input_features()

st.subheader('Fitur Input Pengguna')
st.write(df)

if st.button('Prediksi'):
    df = df[['ca', 'oldpeak', 'exang', 'thal', 'sex', 'age', 'slope', 'cp', 'thalach']]
    df_scaled = scaler.transform(df)
    
    prediction_proba = model.predict_proba(df_scaled)
    prediction = (prediction_proba[:, 1] >= THRESHOLD).astype(int)

    st.subheader('Hasil Prediksi')
    if prediction[0] == 1:
        st.write('Terdeteksi Penyakit Jantung')
        st.subheader('Probabilitas Prediksi')
        st.write(f"Probabilitas: {prediction_proba[0][1]:.2f}")
    else:
        st.write('Tidak Terdeteksi Penyakit Jantung')
        st.subheader('Probabilitas Prediksi')
        st.write(f"Probabilitas: {1 - prediction_proba[0][1]:.2f}")

st.markdown(
    """
    <div class="footer">
        Jaga Kesehatan
    </div>
    """,
    unsafe_allow_html=True
)
