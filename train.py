import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from data_preprocess import load_and_clean_data
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, model_name, threshold):
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted NO", "Predicted YES"],
        yticklabels=["Actual NO", "Actual YES"]
    )

    plt.title(f"Confusion Matrix\nModel: {model_name.upper()}, Threshold: {threshold}")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = f"confusion_matrix_{model_name}_thr_{threshold:.2f}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {filename}")



# Ladowanie i Preprocessing

DATA_PATH = "data/lung_cancer_dataset.csv"

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


# Modelowanie

models = {
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
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


# Wybor modelu na podstawie roc auc
print("\n====Wybor modelu (ROC-AUC):")

roc_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)

    roc_results[name] = (roc, model)

    print(f"{name.upper():6s} ROC-AUC = {roc:.4f}")

best_model_name, (best_roc, best_model) = max(
    roc_results.items(), key=lambda x: x[1][0]
)

print(f"\n====Najlepszy model: {best_model_name.upper()} (ROC-AUC = {best_roc:.4f})====")


# Dosotowanie z progiem - threshold

print("\n====THRESHOLD - wybór la minimalnego FN====")

thresholds = np.arange(0.48, 0.58, 0.02)

y_proba = best_model.predict_proba(X_test)[:, 1]

best_threshold = None
min_fn = float("inf")

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(
        f"\nThreshold = {t:.2f} | "
        f"FN = {fn} | "
        f"Recall = {rec:.4f} | "
        f"Precision = {prec:.4f} | "
        f"F1 = {f1:.4f}"
    )
    print("Confusion Matrix:")
    print(cm)

    if fn < min_fn:
        min_fn = fn
        best_threshold = t


print(f"\nWybrany threshold: {best_threshold:.2f} (min FN = {min_fn})")

final_y_pred = (y_proba >= best_threshold).astype(int)
final_cm = confusion_matrix(y_test, final_y_pred)

plot_confusion_matrix(final_cm, best_model_name, best_threshold)



# Zapisanie finalnego modelu

joblib.dump(
    {
        "model": best_model,
        "features": X.columns.tolist(),
        "threshold": best_threshold
    },
    "model.joblib"
)

print("\nModel zapisany do pliku model.joblib")






