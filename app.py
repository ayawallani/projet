import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

# ======================
# CONFIGURATION PAGE
# ======================
st.set_page_config(page_title="Dashboard Sales", layout="wide")

st.title("📊 Dashboard Sales Forecasting")

# ======================
# UPLOAD DATA
# ======================
st.sidebar.header("📁 Upload des données")

file = st.sidebar.file_uploader("Choisir un fichier CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # ======================
    # APERÇU DES DONNÉES
    # ======================
    st.subheader("🔍 Aperçu des données")
    st.dataframe(df.head())

    # ======================
    # STATISTIQUES
    # ======================
    st.subheader("📈 Statistiques descriptives")
    st.write(df.describe())

    # ======================
    # FILTRES
    # ======================
    st.sidebar.header("🎯 Filtres")

    col = st.sidebar.selectbox("Choisir une colonne", df.columns)
    unique_values = df[col].unique()
    selected_value = st.sidebar.selectbox("Choisir une valeur", unique_values)

    filtered_df = df[df[col] == selected_value]

    st.subheader("📊 Données filtrées")
    st.dataframe(filtered_df)

    # ======================
    # VISUALISATIONS
    # ======================
    st.subheader("📊 Visualisations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Histogramme")
        num_col = st.selectbox("Choisir une variable numérique", df.select_dtypes(include=np.number).columns)
        fig = px.histogram(df, x=num_col)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("### Boxplot")
        fig2 = px.box(df, y=num_col)
        st.plotly_chart(fig2, use_container_width=True)

    # ======================
    # CORRELATION
    # ======================
    st.subheader("🔥 Heatmap des corrélations")

    corr = df.corr(numeric_only=True)
    fig3 = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig3, use_container_width=True)

    # ======================
    # MACHINE LEARNING
    # ======================
    st.subheader("🤖 Prédiction")

    try:
        model = joblib.load("model.pkl")

        st.write("Entrer les valeurs :")

        inputs = []
        for col in df.select_dtypes(include=np.number).columns:
            val = st.number_input(f"{col}", value=0.0)
            inputs.append(val)

        if st.button("🔮 Prédire"):
            prediction = model.predict([inputs])
            st.success(f"Résultat : {prediction[0]}")

    except:
        st.warning("⚠️ Aucun modèle trouvé (model.pkl)")

else:
    st.info("👈 Upload un fichier CSV pour commencer")