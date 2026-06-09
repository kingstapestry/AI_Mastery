import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==================== 1. LOAD DATASET ====================
"""
We'll use a popular movie reviews dataset for sentiment analysis (positive/negative).
"""

url = "https://raw.githubusercontent.com/laxmimerit/IMDB-Movie-Reviews-Sentiment-Analysis/master/IMDB-Dataset.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print(df.head())
print("\nSentiment Distribution:\n", df['sentiment'].value_counts())


# ==================== 2. DATA PREPROCESSING ====================
# TODO: Convert sentiment labels to numeric (positive = 1, negative = 0)


# ==================== 3. TRAIN-TEST SPLIT ====================
# TODO: Split data (test_size=0.2, random_state=42, stratify=y)


# ==================== 4. CREATE PIPELINE ====================
# TODO: Create a Pipeline with:
# 1. TfidfVectorizer (converts text to numbers)
# 2. RandomForestClassifier (or LogisticRegression)


# ==================== 5. TRAIN AND EVALUATE ====================
# TODO: Fit the pipeline, make predictions, and print:
# - Accuracy
# - Classification Report
# - Confusion Matrix


# ==================== 6. TRY DIFFERENT MODELS ====================
# TODO: Create another pipeline using LogisticRegression and compare results


# ==================== 7. FEATURE IMPORTANCE (Advanced) ====================
# TODO: Show the most important words/features for positive and negative sentiment


# ==================== EXERCISES ====================
# 1. Try different ngram_range in TfidfVectorizer (e.g., (1,2) for bigrams)
# 2. Add stop_words='english' to TfidfVectorizer
# 3. Try a more advanced model (e.g., with class_weight)