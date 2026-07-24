from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/random_forest.pkl")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        news = request.form["news"]

        news = preprocess_text(news)

        news_vector = vectorizer.transform([news])

        pred = model.predict(news_vector)[0]
        prob = model.predict_proba(news_vector)[0]

        if pred == 1:
            prediction = "REAL NEWS"
            confidence = round(prob[1] * 100, 2)
        else:
            prediction = "FAKE NEWS"
            confidence = round(prob[0] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)