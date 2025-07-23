import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from category_encoders import BinaryEncoder
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="ML Model Prediction App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi
st.title("🤖 Prediksi Langganan Deposito Berjangka")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi", "Informasi Model", "Visualisasi Data"])

# Fungsi untuk load model
@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as file:
            model_dict = pickle.load(file)
            model = model_dict["model"]
            threshold = model_dict.get("threshold", 0.45)  # Default ke 0.45 jika tidak ada
        return model, threshold
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Fungsi untuk load preprocessor
@st.cache_resource
def load_preprocessor():
    try:
        transformer = joblib.load('transformer.joblib')
        st.success("Preprocessor berhasil dimuat!")
        return transformer
    except FileNotFoundError:
        st.error("File 'transformer.joblib' tidak ditemukan! Pastikan file ada di direktori yang sama.")
        return None

# Fungsi untuk preprocessing data batch
def preprocess_data(df, transformer, expected_columns):
    try:
        # Periksa apakah semua kolom yang diharapkan ada
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Kolom berikut hilang di file CSV: {missing_cols}")
            return None

        # Pastikan urutan kolom sesuai
        df = df[expected_columns]

        # Transformasi data menggunakan transformer
        processed_data = transformer.transform(df)
        return processed_data
    except Exception as e:
        st.error(f"Error saat preprocessing data: {e}")
        return None

# Load model dan preprocessor
model, threshold = load_model()
transformer = load_preprocessor()

# Daftar fitur numerik dan kategorikal
numeric_features = ['age', 'campaign', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx']
categorical_features = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 
                       'month', 'day_of_week', 'previous', 'poutcome', 'pdays_group']
expected_columns = numeric_features + categorical_features

# Nilai default untuk dropdown
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
               'retired', 'self-employed', 'services', 'student', 'technician', 
               'unemployed']
marital_options = ['married', 'single', 'divorced']
education_options = ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 
                    'professional.course', 'university.degree', 'illiterate']
housing_options = ['no', 'yes']
loan_options = ['no', 'yes']
contact_options = ['telephone', 'cellular']
month_options = ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep']
day_of_week_options = ['mon', 'tue', 'wed', 'thu', 'fri']
previous_options = ['not_contacted', 'contacted']
poutcome_options = ['nonexistent', 'failure', 'success']
pdays_group_options = ['never_contacted', 'contacted_before', 'recent']

if page == "Prediksi":
    st.header("🔮 Prediksi")
    
    if model is None or transformer is None:
        st.error("Model atau preprocessor tidak berhasil dimuat. Pastikan file model dan transformer tersedia.")
    else:
        st.success("Model dan preprocessor berhasil dimuat!")
        
        # Form input untuk prediksi tunggal
        st.subheader("Masukkan Data Klien (Prediksi Tunggal)")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📋 Informasi Pribadi")
            age = st.number_input("Umur (Age)", min_value=17, max_value=69, value=30)
            job = st.selectbox("Pekerjaan (Job)", options=job_options)
            marital = st.selectbox("Status Perkawinan (Marital)", options=marital_options)
            education = st.selectbox("Pendidikan (Education)", options=education_options)
            housing = st.selectbox("Pinjaman Perumahan (Housing)", options=housing_options)
            loan = st.selectbox("Pinjaman Pribadi (Loan)", options=loan_options)

        with col2:
            st.markdown("### 📞 Informasi Kampanye")
            contact = st.selectbox("Jenis Kontak (Contact)", options=contact_options)
            month = st.selectbox("Bulan Kontak (Month)", options=month_options)
            day_of_week = st.selectbox("Hari Kontak (Day of Week)", options=day_of_week_options)
            campaign = st.number_input("Jumlah Kontak Kampanye Ini (Campaign)", min_value=1, max_value=6, value=1)
            previous = st.selectbox("Kontak Sebelumnya (Previous)", options=previous_options)
            poutcome = st.selectbox("Hasil Kampanye Sebelumnya (Poutcome)", options=poutcome_options)
            pdays_group = st.selectbox("Kelompok Pdays (Pdays_group)", options=pdays_group_options)

        with col3:
            st.markdown("### 🌍 Informasi Ekonomi Dunia")
            emp_var_rate = st.number_input("Tingkat Variasi Ketenagakerjaan (Emp.var.rate)", 
                                        min_value=-3.4, max_value=1.4, value=1.1, step=0.1)
            cons_price_idx = st.number_input("Indeks Harga Konsumen (Cons.price.idx)", 
                                            min_value=92.201, max_value=94.767, value=93.994, step=0.001)
            cons_conf_idx = st.number_input("Indeks Kepercayaan Konsumen (Cons.conf.idx)", 
                                            min_value=-50.8, max_value=-26.95, value=-36.4, step=0.1)

        # Prediksi tunggal
        if st.button("🚀 Prediksi Tunggal", type="primary"):
            try:
                # Siapkan data input
                input_data = pd.DataFrame({
                    'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
                    'housing': [housing], 'loan': [loan], 'contact': [contact], 'month': [month],
                    'day_of_week': [day_of_week], 'campaign': [campaign], 'previous': [previous],
                    'poutcome': [poutcome], 'pdays_group': [pdays_group], 'emp.var.rate': [emp_var_rate],
                    'cons.price.idx': [cons_price_idx], 'cons.conf.idx': [cons_conf_idx]
                })

                # Preprocessing
                X_input = transformer.transform(input_data)

                # Prediksi
                prob = model.predict_proba(X_input)[:, 1]
                prediction = (prob >= threshold).astype(int)
                prediction_label = "Yes" if prediction[0] == 1 else "No"

                # Tampilkan hasil
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediksi", prediction_label)
                with col2:
                    st.metric("Probabilitas", f"{prob[0]:.2%}")
                with col3:
                    status = "Tinggi" if prob[0] > 0.7 else ("Sedang" if prob[0] > 0.4 else "Rendah")
                    st.metric("Tingkat Kepercayaan", status)

                # Visualisasi probabilitas
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob[0],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilitas Berlangganan"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.4], 'color': "lightgray"},
                            {'range': [0.4, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error membuat prediksi: {e}")

        # Upload file untuk batch prediction
        st.subheader("📁 Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV file untuk prediksi batch", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview data:")
                st.dataframe(df.head())

                if st.button("Prediksi Batch"):
                    # Preprocessing data
                    processed_data = preprocess_data(df, transformer, expected_columns)
                    if processed_data is not None:
                        # Prediksi probabilitas
                        probabilities = model.predict_proba(processed_data)[:, 1]
                        predictions = (probabilities >= threshold).astype(int)
                        prediction_labels = ["Yes" if pred == 1 else "No" for pred in predictions]

                        # Tambahkan hasil ke DataFrame
                        df['Prediction'] = prediction_labels
                        df['Probability'] = probabilities.round(3)

                        st.success("Prediksi batch berhasil!")
                        st.dataframe(df)

                        # Download hasil prediksi
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Hasil Prediksi",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")

elif page == "Informasi Model":
    st.header("📊 Informasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detail Model")
        st.info("""
        **Tipe Model**: CatBoost Classifier  
        **Tanggal Pelatihan**: 23 July 2025  
        **Threshold**: 0.45  
        **Fitur**: 16 fitur  
        **Kelas**: 2 (Yes/No untuk langganan deposito)
        """)
        
        st.subheader("Pentingnya Fitur")
        # Ambil feature importance dari model CatBoost
        try:
            feature_importance = model.get_feature_importance()
            # Nama fitur setelah transformasi (dari ColumnTransformer)
            feature_names = (['education'] + 
                            [f"{col}_{val}" for col in ['marital', 'housing', 'loan', 'contact', 'poutcome', 'previous', 'pdays_group'] 
                             for val in transformer.named_transformers_['onehot'].get_feature_names_out([col])] +
                            transformer.named_transformers_['binary'].get_feature_names_out(['job', 'month', 'day_of_week']) +
                            numeric_features)
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(feature_importance)],
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            
            fig = px.bar(
                importance_df.head(10), 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="Pentingnya Fitur (Top 10)",
                color='Importance',
                color_continuous_scale="viridis"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Tidak dapat menampilkan feature importance. Pastikan model sudah dimuat.")

    with col2:
        st.subheader("Metrik Performa")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Akurasi", f"{0.88:.2f}", "Sesuaikan")
            st.metric("Presisi", f"{0.47:.2f}", "Sesuaikan")
        with metrics_col2:
            st.metric("Recall", f"{0.54:.2f}", "Sesuaikan")
            st.metric("F1-Score", f"{0.51:.2f}", "Sesuaikan")
        
        st.subheader("Confusion Matrix")
        # Matriks kebingungan (misalnya, dalam persentase atau jumlah absolut)
        conf_matrix = [[92.36, 7.64], [45.69, 54.31]]  # Asumsi persentase
        fig = px.imshow(
            conf_matrix,
            text_auto=True,
            aspect=1,
            title="Confusion Matrix",
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah (%)"),  # Tambahkan unit jika persentase
            x=['No', 'Yes'],
            y=['No', 'Yes']
        )
        fig.update_layout(
            width=650,  # Lebar tetap
            height=500,  # Tinggi tetap untuk persegi
            margin=dict(l=75, r=75, t=50, b=50)  # Tambahkan margin untuk mencegah clipping
        )
        st.plotly_chart(fig, use_container_width=False)  # Matikan auto-width

elif page == "Visualisasi Data":
    st.header("📈 Visualisasi Data")
    
    # Load data asli
    try:
        data = pd.read_csv("data-model.csv")
        st.write("Preview data asli:")
        st.dataframe(data.head())
    except FileNotFoundError:
        st.error("File 'data-model.csv' tidak ditemukan! Pastikan file ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error memuat data: {e}")
        st.stop()

    st.subheader("Distribusi Data")

    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram untuk age
        fig = px.histogram(
            data,
            x='age',
            color='deposit',
            title="Distribusi Umur berdasarkan Deposit",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot untuk campaign
        fig = px.box(
            data,
            x='pdays_group',  # Pastikan pdays_group ada di dataset atau tambahkan logika konversi
            y='campaign',
            color='deposit',
            title="Jumlah Kontak Kampanye berdasarkan Pdays Group dan Deposit"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot untuk emp.var.rate vs cons.price.idx
        fig = px.scatter(
            data,
            x='emp.var.rate',
            y='cons.price.idx',
            color='deposit',
            title="Emp.var.rate vs Cons.price.idx"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart untuk distribusi deposit
        target_counts = data['deposit'].value_counts()
        # Peta ulang indeks 0 dan 1 ke No dan Yes
        deposit_mapping = {0: 'No', 1: 'Yes'}
        target_counts.index = target_counts.index.map(deposit_mapping.get)
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="Distribusi Deposit"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit</p>
        <p>© 2025 Purwadhika</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Cara Penggunaan:**
    1. Pilih halaman dari navigasi
    2. Masukkan fitur untuk prediksi tunggal atau unggah CSV untuk prediksi batch
    3. Klik 'Prediksi' untuk melihat hasil
    4. Lihat informasi model dan visualisasi data
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Kontak:**")
st.sidebar.markdown("1. Dimas Maulidin Firdaus")
st.sidebar.markdown("- dhymas.maulidin@gmail.com")
st.sidebar.markdown("- [LinkedIn](https://www.linkedin.com/in/dhymasmf/)" )
st.sidebar.markdown("- [Github](https://github.com/dhymasmf)")
st.sidebar.markdown("2. M. Rikza N. Fachry")
st.sidebar.markdown("- fachry240702@gmail.com")
st.sidebar.markdown("- [LinkedIn](www.linkedin.com/in/mrikzanf)")
st.sidebar.markdown("- [Gituhub](https://github.com/Rick-zaa)")
