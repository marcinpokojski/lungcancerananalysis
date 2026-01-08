import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from preprocess import load_and_clean_data

#path for dataset
DATA_PATH = "data/lung_cancer_dataset.csv"


if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_clean_data(DATA_PATH)

    # Train / test split (80 training, 20 test)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["lung_cancer"]
    )

    # train model
    predictor = TabularPredictor(
        label="lung_cancer",
        problem_type="binary",
        eval_metric="roc_auc"
    ).fit(
        train_df,
        presets="medium_quality",
        hyperparameters={
            "GBM": {},
            "CAT": {},
            "NN_TORCH": {},
        }
    )

    #evaluate model
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )

    y_true = test_df["lung_cancer"]
    X_test = test_df.drop(columns=["lung_cancer"])

    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)[1]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("Model trained and saved automatically by AutoGluon.")

    leaderboard = predictor.leaderboard(test_df, silent=True)
    print(leaderboard)
