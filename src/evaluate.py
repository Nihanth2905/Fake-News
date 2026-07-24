import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

os.makedirs("reports", exist_ok=True)

df = pd.read_csv("data/processed.csv")

X = df["content"]
y = df["label"]

vectorizer = joblib.load("models/vectorizer.pkl")

X = vectorizer.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "KNN": "models/knn.pkl",
    "Random Forest": "models/random_forest.pkl",
    "MLPClassifier": "models/mlpclassifier.pkl"
}

results = []

for name, path in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model = joblib.load(path)

    try:
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    except Exception as e:
        print(f"Skipped: {e}")

results_df = pd.DataFrame(results)

results_df.to_csv("reports/results.csv", index=False)

print("\nResults saved successfully!")
print("Location: reports/results.csv")