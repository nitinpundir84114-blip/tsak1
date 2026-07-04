# ================================
# Movie Genre Classification
# CodSoft Task 1
# ================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("Loading Dataset...")

# Load Dataset
df = pd.read_csv(
    "Genre Classification Dataset/train_data.txt",
    sep=" ::: ",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"],
    engine="python"
)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Features and Target
X = df["DESCRIPTION"]
y = df["GENRE"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Model...")

# Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\n==========================")
print("Accuracy")
print("==========================")
print(accuracy_score(y_test, y_pred))

print("\n==========================")
print("Classification Report")
print("==========================")
print(classification_report(y_test, y_pred))

# ----------------------------------
# Predict New Movie Genre
# ----------------------------------

while True:

    text = input("\nEnter Movie Description (type exit to quit): ")

    if text.lower() == "exit":
        break

    prediction = model.predict([text])

    print("Predicted Genre :", prediction[0])