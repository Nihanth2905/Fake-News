import pandas as pd

print("Loading datasets...")

fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

print("Datasets loaded successfully!")

fake_df["label"] = 0
true_df["label"] = 1

news_df = pd.concat([fake_df, true_df], ignore_index=True)

print("\nTotal rows after merging:", len(news_df))

news_df.drop(["subject", "date"], axis=1, inplace=True)

news_df.dropna(inplace=True)

news_df.drop_duplicates(inplace=True)

news_df["content"] = news_df["title"] + " " + news_df["text"]

news_df = news_df[["content", "label"]]

news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)

news_df.to_csv("data/processed.csv", index=False)

print("\nPreprocessing Completed Successfully!")

print("\nDataset Shape:")
print(news_df.shape)

print("\nFirst 5 Rows:")
print(news_df.head())

print("\nLabel Distribution:")
print(news_df["label"].value_counts())

print("\nProcessed dataset saved as data/processed.csv")