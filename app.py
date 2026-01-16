import streamlit as st
import pandas as pd
import joblib
import os

from data_preprocess import load_and_clean_data

MODEL_PATH = "model.joblib"

st.set_page_config(page_title="Cancer Predict", layout="centered")

@st.cache_resource
def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"], data["features"]

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found")
    st.stop()

predictor, feature_columns = load_model()

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

col1, col2 = st.columns(2)
with col1:
    if st.button("Home", use_container_width=True):
        st.session_state["page"] = "Home"
with col2:
    if st.button("Prediction", use_container_width=True):
        st.session_state["page"] = "Prediction"

st.divider()

if st.session_state["page"] == "Home":
    st.title("Lung Cancer Risk Estimation Tool")
    st.write("Model file:", "model.joblib")
    st.write("Model type:", type(predictor).__name__)

    st.divider()

    st.markdown("""
        <div style="border:1px solid #444; padding:15px; border-radius:10px; margin-bottom:15px;">
            <h3>O Nas</h3>
            Franciszek Gruszecki s27619 <br>
            Marcin Pokojsku s26779
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="border:1px solid #444; padding:15px; border-radius:10px; margin-bottom:15px;">
            <h3>O projekcie</h3>
            Aplikacja służy do szacowania ryzyka raka płuc na podstawie podstawowych
            czynników zdrowotnych i środowiskowych z wykorzystaniem modelu uczenia maszynowego.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="border:1px solid #444; padding:15px; border-radius:10px; margin-bottom:15px;">
            <h3>Dataset</h3>
            Dataset pochodzi z serwisu Kaggle i zawiera dane dotyczące czynników ryzyka raka płuc.
            <br>
            <a href="https://www.kaggle.com/code/finlaymcandrew/lung-cancer-risk-analysis/input" target="_blank">
                Link do datasetu
            </a>
        </div>
        """, unsafe_allow_html=True)

if st.session_state["page"] == "Prediction":
    st.title("Cancer Risk Prediction")

    age = st.slider("Age", 18, 100, 50)
    gender = st.selectbox("Sex", ["male", "female"])
    pack_years = st.slider("Pack years", 0, 100, 0)
    radon_exposure = st.selectbox("Radon exposure", ["low", "medium", "high"])
    asbestos_exposure = st.checkbox("Asbestos exposure")
    secondhand_smoke_exposure = st.checkbox("Secondhand smoke exposure")
    copd_diagnosis = st.checkbox("COPD diagnosis")
    alcohol_consumption = st.selectbox("Alcohol consumption", ["none", "moderate", "heavy"])
    family_history = st.checkbox("Family history")

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "age": age,
            "pack_years": pack_years,
            "gender": gender,
            "radon_exposure": radon_exposure,
            "asbestos_exposure": "yes" if asbestos_exposure else "no",
            "secondhand_smoke_exposure": "yes" if secondhand_smoke_exposure else "no",
            "copd_diagnosis": "yes" if copd_diagnosis else "no",
            "alcohol_consumption": alcohol_consumption,
            "family_history": "yes" if family_history else "no"
        }])

        input_df = load_and_clean_data(input_df)

        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        prediction = predictor.predict(input_df)[0]
        probability = predictor.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"YES ({probability:.1%})")
        else:
            st.success(f"NO")
