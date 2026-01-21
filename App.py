import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Cluster Shoppers", layout="wide")

# --- Judul dan Deskripsi ---
st.title("🛍️ Clustering Pengunjung Online Shop")
st.markdown("""
Aplikasi ini menggunakan **K-Means Clustering** untuk mengelompokkan pengunjung website 
berdasarkan perilaku browsing mereka. Masukkan data di sidebar untuk memprediksi profil pengunjung baru.
""")

# --- 1. Load & Train Data ---
@st.cache_data # Cache agar tidak perlu training ulang setiap kali input berubah
def load_and_train_model():
    try:
        # Coba baca file lokal, pastikan file csv ada di satu folder dengan app.py
        df = pd.read_csv('online_shoppers_intention.csv')
    except FileNotFoundError:
        st.error("File 'online_shoppers_intention.csv' tidak ditemukan. Harap upload file atau taruh di folder yang sama.")
        return None, None, None, None

    # Fitur yang digunakan (Sesuai kode colab sebelumnya)
    features = ['Administrative', 'Administrative_Duration', 
                'Informational', 'Informational_Duration', 
                'ProductRelated', 'ProductRelated_Duration', 
                'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
    
    X = df[features].fillna(0) # Handle missing values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modeling (K=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Gabungkan label ke dataframe asli untuk visualisasi
    df['Cluster'] = kmeans.labels_
    
    return df, scaler, kmeans, features

# Memuat Model
df, scaler, kmeans_model, feature_names = load_and_train_model()

if df is not None:
    # --- 2. Sidebar: Input User ---
    st.sidebar.header("🔍 Input Data Pengunjung Baru")
    st.sidebar.write("Masukkan parameter perilaku pengunjung:")

    def user_input_features():
        # Membuat input number untuk setiap fitur
        admin = st.sidebar.number_input("Administrative (Halaman Admin dilihat)", min_value=0, value=0)
        admin_dur = st.sidebar.number_input("Administrative Duration (Detik)", min_value=0.0, value=0.0)
        
        info = st.sidebar.number_input("Informational (Halaman Info dilihat)", min_value=0, value=0)
        info_dur = st.sidebar.number_input("Informational Duration (Detik)", min_value=0.0, value=0.0)
        
        prod = st.sidebar.number_input("ProductRelated (Halaman Produk dilihat)", min_value=0, value=1)
        prod_dur = st.sidebar.number_input("ProductRelated Duration (Detik)", min_value=0.0, value=10.0)
        
        bounce = st.sidebar.slider("BounceRates (0 - 1)", 0.0, 1.0, 0.01)
        exit_r = st.sidebar.slider("ExitRates (0 - 1)", 0.0, 1.0, 0.02)
        
        page_val = st.sidebar.number_input("PageValues (Nilai Halaman)", min_value=0.0, value=0.0)
        special = st.sidebar.slider("SpecialDay (Kedekatan hari raya 0-1)", 0.0, 1.0, 0.0)

        data = {
            'Administrative': admin,
            'Administrative_Duration': admin_dur,
            'Informational': info,
            'Informational_Duration': info_dur,
            'ProductRelated': prod,
            'ProductRelated_Duration': prod_dur,
            'BounceRates': bounce,
            'ExitRates': exit_r,
            'PageValues': page_val,
            'SpecialDay': special
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # Tampilkan input user
    st.subheader("Data Pengunjung yang Dimasukkan:")
    st.write(input_df)

    # --- 3. Prediksi ---
    if st.button("Prediksi Cluster"):
        # Scale data input menggunakan scaler yang sama dengan data latih
        input_scaled = scaler.transform(input_df)
        
        # Prediksi
        cluster_prediction = kmeans_model.predict(input_scaled)[0]
        
        st.success(f"Pengunjung ini termasuk dalam **Cluster {cluster_prediction}**")

        # Interpretasi sederhana (Contoh interpretasi, sesuaikan dengan hasil analisis Anda)
        if cluster_prediction == 0:
            st.info("Karakteristik: Mungkin pengunjung biasa/window shopper.")
        elif cluster_prediction == 1:
            st.info("Karakteristik: Pengunjung sangat aktif atau tertarik.")
        else:
            st.info("Karakteristik: Pengunjung dengan potensi transaksi tinggi.")

        # --- 4. Visualisasi ---
        st.subheader("Visualisasi Posisi Data Baru")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter Plot 1: Product Related vs Duration
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x='ProductRelated', y='ProductRelated_Duration', hue='Cluster', palette='viridis', alpha=0.6, ax=ax)
            # Plot titik data baru
            ax.scatter(input_df['ProductRelated'], input_df['ProductRelated_Duration'], color='red', s=200, marker='*', label='Data Baru')
            ax.legend()
            ax.set_title("Product Views vs Duration")
            st.pyplot(fig)

        with col2:
            # Scatter Plot 2: ExitRates vs PageValues (Penting untuk konversi)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x='ExitRates', y='PageValues', hue='Cluster', palette='viridis', alpha=0.6, ax=ax2)
            # Plot titik data baru
            ax2.scatter(input_df['ExitRates'], input_df['PageValues'], color='red', s=200, marker='*', label='Data Baru')
            ax2.legend()
            ax2.set_title("Exit Rates vs Page Values")
            st.pyplot(fig2)

else:
    st.warning("Silakan upload dataset 'online_shoppers_intention.csv' ke folder aplikasi untuk memulai.")
