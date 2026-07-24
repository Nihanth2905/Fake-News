import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from models import get_models

# Load processed dataset
df = pd.read_csv("data/processed.csv")

X = df["content"]
y = df["label"]

# TF-IDF Vectorization (Optimized)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    dtype="float32"
)

X = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

# Load models
models = get_models()

# Train and Save Models
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, f"models/{filename}")

    accuracy = model.score(X_test, y_test)

    print(f"{name} Accuracy: {accuracy:.4f}")

print("\nAll models trained and saved successfully!")