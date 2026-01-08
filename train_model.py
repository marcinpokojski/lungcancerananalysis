import os
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from preprocess import load_and_clean_data

DATA_PATH = "data/lung_cancer_dataset.csv"
MODEL_DIR = "AutogluonModels"


import shutil

def model_exists():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    return False



def train_and_save_model():
    df = load_and_clean_data(DATA_PATH)

    train_df, _ = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["lung_cancer"]
    )

    predictor = TabularPredictor(
        label="lung_cancer",
        problem_type="binary",
        eval_metric="roc_auc",
        path=MODEL_DIR
    ).fit(
        train_df,
        presets="medium_quality",
        hyperparameters={
            "GBM": {},
            "CAT": {},
            "NN_TORCH": {},
        }
    )

    return predictor
