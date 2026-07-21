import pandas as pd
import joblib
data = pd.read_csv("comments.csv")

print(data)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
X = data["comment"]
y = data["label"]
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)
joblib.dump(model, "model.pkl")

print("Model Saved Successfully!")

print("AI Model Trained Successfully!")