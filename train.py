import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from preprocess import load_and_clean_data

DATA_PATH = "data/lung_cancer_dataset.csv"

# load + preprocess
df = load_and_clean_data(DATA_PATH)

X = df.drop(columns=["lung_cancer"])
y = df["lung_cancer"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# models (pipelines)
models = {
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"  # important for medical data
        ))
    ]),

    "rf": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),

    "gb": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
}

results = {}

# train + evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)

    results[name] = (roc, model)
    print(f"{name}: ROC-AUC = {roc:.4f}")

# pick best
best_name, (best_roc, best_model) = max(
    results.items(),
    key=lambda x: x[1][0]
)

print(f"\nBest model: {best_name} (ROC-AUC={best_roc:.4f})")

# save model + feature order
joblib.dump(
    {
        "model": best_model,
        "features": X.columns.tolist()
    },
    "model.joblib"
)

print("Model saved to model.joblib")
