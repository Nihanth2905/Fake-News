import re
import joblib

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/random_forest.pkl")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("=" * 50)
print("AI-Powered Fake News Detection")
print("=" * 50)

while True:

    news = input("\nEnter News (or type 'exit' to quit): ")

    if news.lower() == "exit":
        print("Exiting...")
        break

    news = preprocess_text(news)

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)[0]
    probability = model.predict_proba(news_vector)[0]

    if prediction == 1:
        confidence = probability[1] * 100
        print(f"\nPrediction : REAL NEWS")
        print(f"Confidence : {confidence:.2f}%")
    else:
        confidence = probability[0] * 100
        print(f"\nPrediction : FAKE NEWS")
        print(f"Confidence : {confidence:.2f}%")