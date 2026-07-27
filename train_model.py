import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
 
# Load data
data = pd.read_csv("comments.csv")
 
# Drop rows with missing comment/label (avoids crashes in TfidfVectorizer)
data = data.dropna(subset=["comment", "label"])
data["comment"] = data["comment"].astype(str)
 
print("Label distribution:")
print(data["label"].value_counts())
print()
 
X = data["comment"]
y = data["label"]
 
# Split into train/test so we can actually measure accuracy,
# instead of training blindly on 100% of the data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
 
model = Pipeline([
    ("vectorizer", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])
 
# Train on the training split
model.fit(X_train, y_train)
 
# Evaluate on the held-out test split
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
 
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))
 
# Retrain on the FULL dataset before saving, so the deployed model
# benefits from all available data (common practice once you've
# validated performance on the split above).
model.fit(X, y)
joblib.dump(model, "model.pkl")
 
print("Model Saved Successfully!")
print("AI Model Trained Successfully!")
 