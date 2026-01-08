import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
import os
from preprocess import load_and_clean_data


st.set_page_config(page_title="Cancer Predict", layout="centered")

st.markdown(
    """
    <style>
        .stButton > button {
            background-color: transparent;
            color: white;
            border: none;
            border-bottom: 2px solid transparent;
            font-size: 16px;
            font-weight: 600;
            padding: 8px 0;
        }

        .stButton > button:hover {
            border-bottom: 2px solid white;
            background-color: transparent;
        }
    </style>
    """,
    unsafe_allow_html=True
)



def get_latest_model_path(base_dir="AutogluonModels"):
    models = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("ag-")
    ]
    if not models:
        raise FileNotFoundError("No AutoGluon models found.")
    return max(models, key=os.path.getmtime)


@st.cache_resource
def load_model():
    model_path = get_latest_model_path()
    return TabularPredictor.load(model_path)


predictor = load_model()


if "page" not in st.session_state:
    st.session_state["page"] = "Home"


tab_home, tab_prediction = st.columns(2)

with tab_home:
    if st.button("Home", use_container_width=True):
        st.session_state["page"] = "Home"

with tab_prediction:
    if st.button("Prediction", use_container_width=True):
        st.session_state["page"] = "Prediction"

page = st.session_state["page"]

st.divider()


if page == "Home":
    st.title("Lung Cancer Risk Estimation Tool")
    st.write("Model path:", predictor.path)
    st.write("Best model name:", predictor.model_best)


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

if page == "Prediction":
    st.title("Cancer Risk Prediction")
    st.subheader("Patient Risk Assessment Form")
    st.divider()

    age = st.slider("Age", 1, 100, 30)
    st.caption("Patient age in years")
    st.divider()

    gender = st.selectbox("Sex", ["male", "female"])
    st.caption("Biological sex of the patient")
    st.divider()

    pack_years = st.slider("Pack years", 0, 100, 0)
    st.caption("Total number of pack years smoked across lifetime")
    st.divider()

    radon_exposure = st.selectbox("Radon exposure", ["low", "medium", "high"])
    st.caption("Estimated level of long-term radon exposure")
    st.divider()

    asbestos_exposure = st.checkbox("Asbestos exposure")
    st.caption("Whether the patient has been exposed to asbestos")
    st.divider()

    secondhand_smoke_exposure = st.checkbox("Secondhand smoke exposure")
    st.caption("Exposure to tobacco smoke from other people")
    st.divider()

    copd_diagnosis = st.checkbox("COPD diagnosis")
    st.caption("Whether the patient has been diagnosed with COPD")
    st.divider()

    alcohol_consumption = st.selectbox("Alcohol consumption", ["none", "moderate", "heavy"])
    st.caption("Level of alcohol consumption")
    st.divider()

    family_history = st.checkbox("Family history")
    st.caption("Family history of lung cancer")
    st.divider()

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "age": age,
            "pack_years": pack_years,
            "asbestos_exposure": int(asbestos_exposure),
            "secondhand_smoke_exposure": int(secondhand_smoke_exposure),
            "copd_diagnosis": int(copd_diagnosis),
            "family_history": int(family_history),
            "alcohol_consumption": (
                0 if alcohol_consumption == "none"
                else 1 if alcohol_consumption == "moderate"
                else 2
            ),
            "radon_exposure": (
                0 if radon_exposure == "low"
                else 1 if radon_exposure == "medium"
                else 2
            ),
            "gender": gender
        }])

        input_df = load_and_clean_data(input_df)
        prediction = predictor.predict(input_df).iloc[0]
        probability = predictor.predict_proba(input_df).iloc[0][1]

        if prediction == 1:
            st.error(f"Model predicts: YES (risk = {probability:.1%})")
        else:
            st.success(f"Model predicts: NO (risk = {probability:.1%})")
