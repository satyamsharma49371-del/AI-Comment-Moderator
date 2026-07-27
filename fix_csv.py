import pandas as pd
 
df = pd.read_csv("comments.csv")
 
# Drop empty rows
df = df.dropna(subset=["comment", "label"])
 
# Normalize labels to lowercase (Safe -> safe, Toxic -> toxic, Spam -> spam)
df["label"] = df["label"].str.strip().str.lower()
 
# Keep only the 3 categories this app actually supports
allowed = ["safe", "toxic", "spam"]
before = len(df)
df = df[df["label"].isin(allowed)]
after = len(df)
 
print(f"Removed {before - after} rows with unsupported labels (kept only safe/toxic/spam).")
print(df["label"].value_counts())
 
df.to_csv("comments.csv", index=False)
print("comments.csv fixed and saved.")
 