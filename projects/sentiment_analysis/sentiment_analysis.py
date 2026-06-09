import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


"""
LESSON 21: Sentiment Analysis on IMDB Movie Reviews

Goal:
    Learn how to work with text data using TF-IDF vectorization — 
    a fundamental skill before moving into modern LLMs.

This is different from previous tabular projects because we are dealing with raw text.
"""

# ==================== 1. LOAD DATASET ====================
url = "https://github.com/Ankit152/IMDB-sentiment-analysis/raw/master/IMDB-Dataset.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print(df.head())
print("\nSentiment Distribution:\n", df['sentiment'].value_counts())


# ==================== 2. DATA PREPROCESSING ====================
# Convert string labels to numbers (required for all ML models)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

print("Labels converted to numeric (1 = positive, 0 = negative)")


# ==================== 3. TRAIN-TEST SPLIT ====================
# X must be the text column only (Series), not a DataFrame
X = df["review"]           # ← Text data
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ==================== 4. CREATE PIPELINE (Random Forest) ====================
"""
TfidfVectorizer is the key step for text:
- Converts raw text into numerical features (importance of each word)
- max_features=5000 limits vocabulary size for speed and performance
"""

rf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000)),
    ('classifier', RandomForestClassifier(random_state=42))
])


# ==================== 5. TRAIN AND EVALUATE ====================
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print("\n=== Random Forest Pipeline Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ==================== 6. LOGISTIC REGRESSION PIPELINE ====================
"""
Logistic Regression often performs surprisingly well on text data with TF-IDF.
"""

lr_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000)),
    ('classifier', LogisticRegression(max_iter=500))
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

print("\n=== Logistic Regression Pipeline Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))


# ==================== 7. MOST IMPORTANT WORDS ====================
print("\n=== Most Important Words for Sentiment ===")

# Extract steps from the better pipeline (Logistic Regression)
vectorizer = lr_pipeline.named_steps['vectorizer']
classifier = lr_pipeline.named_steps['classifier']

feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

word_importance = pd.DataFrame({
    'Word': feature_names,
    'Importance': coefficients
})

print("Top 15 Positive Words:")
print(word_importance.nlargest(15, 'Importance')[['Word', 'Importance']])

print("\nTop 15 Negative Words:")
print(word_importance.nsmallest(15, 'Importance')[['Word', 'Importance']])


# ==================== SUMMARY ====================
"""
Key Takeaways from Text Classification:
- Text must be converted to numbers using TfidfVectorizer (or CountVectorizer)
- Logistic Regression + TF-IDF is often very strong for sentiment analysis
- We can interpret the model by looking at important words
- This is the foundation for more advanced NLP and LLM applications
"""