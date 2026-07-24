import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

os.makedirs("reports", exist_ok=True)

results = pd.read_csv("reports/results.csv")

plt.figure(figsize=(8,5))
plt.bar(results["Model"], results["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("reports/accuracy_comparison.png")
plt.close()

df = pd.read_csv("data/processed.csv")

fake_text = " ".join(df[df["label"] == 0]["content"])
real_text = " ".join(df[df["label"] == 1]["content"])

fake_wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(fake_text)

plt.figure(figsize=(10,5))
plt.imshow(fake_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Fake News Word Cloud")
plt.tight_layout()
plt.savefig("reports/wordcloud_fake.png")
plt.close()

real_wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(real_text)

plt.figure(figsize=(10,5))
plt.imshow(real_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Real News Word Cloud")
plt.tight_layout()
plt.savefig("reports/wordcloud_real.png")
plt.close()

print("Visualizations generated successfully!")
print("Saved in reports folder.")