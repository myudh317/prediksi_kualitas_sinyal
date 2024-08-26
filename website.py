import streamlit as st
import pandas as pd
import folium
from streamlit_option_menu import option_menu
import joblib
from sklearn.preprocessing import StandardScaler
from streamlit_folium import folium_static
from google.cloud import storage
from google.oauth2 import service_account
import numpy as np
from io import BytesIO
import os
from st_files_connection import FilesConnection

# Hide Streamlit style
hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

st.set_page_config(
    page_title="Prediksi Kualitas Sinyal",
    page_icon="memo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/benedictus-briatore-ananta-ba921b281/',
        'Report a bug': "https://github.com/benedictusbriatoreananta/dashboard",
        'About': "## A 'Badspot Prediction Tool' by Benedictus Briatore Ananta"
    }
)

st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    """, unsafe_allow_html=True
)

selected = "Menu Utama"

with st.sidebar:
    selected = option_menu(
        menu_title="Prediksi Kualitas Sinyal",
        options=["Menu Utama", "Predictions", "Contributors"],
        icons=["house", "upload", "people"],
        menu_icon="broadcast tower",
        default_index=0,
    )

@st.cache_resource
def load_models_from_gcs(bucket_name, k_means_path, rf_model_path, scaler_path):
    try:
        # Set up a connection to GCS using FilesConnection
        conn = st.connection('gcs', type=FilesConnection)

        # Load KMeans model directly into memory
        with conn.open(f"{bucket_name}/{k_means_path}", mode="rb") as f:
            k_means_model = joblib.load(BytesIO(f.read()))

        # Load RandomForest model directly into memory
        with conn.open(f"{bucket_name}/{rf_model_path}", mode="rb") as f:
            rf_model = joblib.load(BytesIO(f.read()))

        # Load Scaler model directly into memory
        with conn.open(f"{bucket_name}/{scaler_path}", mode="rb") as f:
            scaler = joblib.load(BytesIO(f.read()))

        return k_means_model, rf_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# =====================================================================================
# Home tab
if selected == "Menu Utama":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            <h1>Prediksi Kualitas Sinyal <i class="fas fa-broadcast-tower"></i></h1>
            """, unsafe_allow_html=True
        )
        st.divider()
        st.header("About :memo:")
        st.markdown('''
        ####
        Selamat datang di situs Prediksi Kualitas Sinyal. Platform kami dirancang untuk lembaga pemerintah, termasuk Kementerian Komunikasi dan Informatika, dalam memprediksi kualitas sinyal di berbagai lokasi, termasuk area di luar rute badspot yang telah diidentifikasi. Dengan memanfaatkan analisis prediktif tingkat lanjut, kami membantu mengidentifikasi wilayah dengan potensi penurunan kualitas sinyal, sehingga memungkinkan intervensi dan dukungan tepat waktu.

        Misi kami adalah mendukung institusi dalam memastikan jaringan komunikasi yang stabil dan andal di seluruh area, termasuk lokasi-lokasi yang kurang terpantau. Kami berkomitmen untuk terus meningkatkan layanan dan menyambut masukan Anda.
        ''')
        
        st.markdown("#### `Get Started Now!`")

# =====================================================================================
# Predictions tab
elif selected == "Predictions":
    st.header("Prediksi Kualitas Sinyal")
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(df)
        
        # Ubah nama kolom agar sesuai dengan yang diharapkan oleh kode
        df.rename(columns={'rsrp (dbm)': 'RSRP'}, inplace=True)
        
        # Convert 'Latitude' and 'Longitude' to float64
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        # Check for NaN values after conversion
        if df[['Latitude', 'Longitude', 'RSRP']].isnull().any().any():
            st.error("Data mengandung nilai NaN di kolom 'Latitude', 'Longitude', atau 'RSRP'.")
            st.write(df[['Latitude', 'Longitude', 'RSRP']])
            st.stop()
        
        # Asumsi data memiliki kolom 'Latitude', 'Longitude', dan 'RSRP'
        if 'Latitude' in df.columns and 'Longitude' in df.columns and 'RSRP' in df.columns:  
            kmeans, rf, scaler = load_models_from_gcs(
                'model-skripsi-ml', 
                'kmeans_model.pkl', 
                'random_forest_model.pkl', 
                'scaler.pkl'
            )
            
            def predict_rsrq(rsrp, longitude, latitude):
                location = np.array([[rsrp, longitude, latitude]])
                location_scaled = scaler.transform(location)
                cluster = kmeans.predict(location_scaled)
                location_with_cluster = np.hstack((location_scaled, cluster.reshape(-1, 1)))
                predicted_rsrq = rf.predict(location_with_cluster)
                return predicted_rsrq[0]

            predictions = []
            for index, row in df.iterrows():
                rsrq_value = predict_rsrq(row['RSRP'], row['Longitude'], row['Latitude'])
                predictions.append(rsrq_value)

            df['rsrq'] = predictions
            st.write("Data dengan prediksi RSRQ:")
            st.write(df)
            
            # Ensure no NaN values in the data used for mapping
            if df[['Latitude', 'Longitude']].isnull().any().any():
                st.error("Data yang digunakan untuk peta mengandung nilai NaN.")
                st.write(df[['Latitude', 'Longitude']])
                st.stop()

            # Load data rute dari Gabungcleaned_2.csv
            route_data = pd.read_csv('Gabung_Cleaned_2.csv')
            route_data.replace('-', np.nan, inplace=True)
            route_data.dropna(subset=['RSRP', 'Longitude', 'Latitude'], inplace=True)

            route_data_subset = route_data.sample(frac=0.1, random_state=42)
            
            # Ensure route_data does not contain NaN values
            if route_data[['Latitude', 'Longitude']].isnull().any().any():
                st.error("Data rute mengandung nilai NaN.")
                st.write(route_data[['Latitude', 'Longitude']])
                st.stop()

            # Create the map
            m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)
            
            # Tambahkan rute dari data Gabungcleaned_2.csv
            for _, row in route_data_subset.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=3,
                    color='green',
                    fill=True,
                    fill_color='green'
                ).add_to(m)
            
            # Tambahkan prediksi data
            for _, row in df.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5,
                    popup=f"RSRP: {row['RSRP']}, RSRQ: {row['rsrq']}",
                    color='blue' if row['rsrq'] > -15 else 'red',
                    fill=True,
                    fill_color='blue' if row['rsrq'] > -15 else 'red'
                ).add_to(m)
            
            folium_static(m)
        else:
            st.error("File Excel harus memiliki kolom 'Latitude', 'Longitude', dan 'RSRP'.")
            st.stop()

            
# =====================================================================================
# Contributors tab
elif selected == "Contributors":
    st.header("Contributors")
    st.markdown('''
    - Benedictus Briatore Ananta
    - [LinkedIn](https://www.linkedin.com/in/benedictus-briatore-ananta-ba921b281/)
    - [GitHub](https://github.com/benedictusbriatoreananta)
    ''')
