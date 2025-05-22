import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import pickle
import matplotlib.pyplot as plt

model = pickle.load(open('model_prediksi_harga.sav', 'rb'))
df1 = pd.read_csv('CarPrice.csv')

st.sidebar.image("bmw.jpg", width=150)
page = st.sidebar.radio("Navigasi", ["🏠 Beranda", "📄 Dataset", "📊 Visualisasi", "📈 Prediksi"])

if page == "🏠 Beranda":
    st.markdown("<h1 style='text-align: center;'>Prediksi Harga Mobil</h1>", unsafe_allow_html=True)
    st.image("bmw.jpg", use_container_width=True)
    st.markdown("""
    <div style="text-align: justify; font-size: 16px;">
        Aplikasi ini menggunakan model <b>Machine Learning</b> untuk memprediksi harga mobil berdasarkan beberapa parameter teknis. 
        Prediksi didasarkan pada fitur:
        <ul>
            <li><code>highway-mpg</code></li>
            <li><code>curb weight</code></li>
            <li><code>horsepower</code></li>
        </ul>
        Gunakan sidebar untuk menjelajah fitur-fitur aplikasi ini.
    </div>
    """, unsafe_allow_html=True)

elif page == "📄 Dataset":
    st.header("📄 Dataset Mobil")
    st.dataframe(df1.style.highlight_max(axis=0, color='lightgreen'))

    with st.expander("ℹ️ Tentang Dataset"):
        st.info("""
        Dataset ini memuat informasi teknis mobil seperti tenaga mesin, berat, dan efisiensi bahan bakar.
        Digunakan sebagai data latih untuk memprediksi harga kendaraan secara otomatis.
        """)

elif page == "📊 Visualisasi":
    st.header("📊 Visualisasi Data")

    st.subheader("📌 Hubungan Horsepower vs Price")
    chart = alt.Chart(df1).mark_circle(size=60).encode(
    x='horsepower',
    y='price',
    tooltip=['horsepower', 'price', 'stroke'],
    color='fueltype'
).interactive()

    st.altair_chart(chart, use_container_width=True)

    st.subheader("📊 Distribusi Symboling")
    if "symboling" in df1.columns:
        fig, ax = plt.subplots()
        df1['symboling'].plot(kind='hist', bins=20, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title("Distribusi Nilai Symboling")
        st.pyplot(fig)
    else:
        st.warning("Kolom 'symboling' tidak ditemukan.")

elif page == "📈 Prediksi":
    st.header("Prediksi Harga Mobil")
    st.markdown("Masukkan parameter berikut untuk melihat estimasi harga:")

    col1, col2 = st.columns(2)
    with col1:
        highwaympg = st.number_input('Highway MPG', min_value=0, step=1)
        horsepower = st.number_input('Horsepower', min_value=0, step=1)
    with col2:
        curbweight = st.number_input('Curb Weight', min_value=0, step=10)

    if st.button('Prediksi'):
        with st.spinner("🔍 Menganalisis..."):
            car_prediction = model.predict([[highwaympg, curbweight, horsepower]])
            harga_mobil = float(car_prediction[0])
            harga_mobil_formatted = f"${harga_mobil:,.2f}"

            st.success("✅ Prediksi Berhasil!")
            st.markdown(f"<h2 style='color: green;'>💵 Harga Mobil: {harga_mobil_formatted}</h2>", unsafe_allow_html=True)

            st.balloons()

st.markdown("---")
st.caption("🚀 Dibuat oleh Hasan Anak Baik | Proyek Machine Learning Web | 2025")
