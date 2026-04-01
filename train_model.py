import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load labeled dataset
df = pd.read_csv("emotion_labeled_dataset.csv")

# Features and labels
X = df["cleaned"]
y = df["emotion"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train SVM Model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Predictions
pred = model.predict(X_test_tfidf)

# Accuracy
acc = accuracy_score(y_test, pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, pred))

# Save model & TF-IDF
joblib.dump(model, "emotion_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

print("\nModel saved successfully!")