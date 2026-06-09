import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score


# ==================== 1. LOAD DATA ====================
"""
Telco Customer Churn Dataset - Very popular real-world business dataset.
Task: Predict which customers will leave the company (churn).
"""

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print(df.head())
print("\nChurn Distribution:\n", df['Churn'].value_counts())


# ==================== 2. DATA CLEANING & PREPROCESSING ====================
# TODO: Drop customerID column
# TODO: Convert TotalCharges to numeric (it has empty strings)
# TODO: Handle missing values
# TODO: Convert Churn column to numeric (Yes=1, No=0)


# ==================== 3. SEPARATE NUMERIC & CATEGORICAL FEATURES ====================
# TODO: Create two lists:
# numeric_features = [...]
# categorical_features = [...]


# ==================== 4. CREATE ADVANCED PIPELINE (ColumnTransformer) ====================
# TODO: Use ColumnTransformer to:
#   - Scale numeric features
#   - OneHotEncode categorical features
# Then add a classifier (RandomForest or LogisticRegression)


# ==================== 5. TRAIN AND EVALUATE ====================
# TODO: Train the pipeline and evaluate with Accuracy, F1, ROC-AUC


# ==================== 6. FEATURE IMPORTANCE ====================
# TODO: Extract and show the most important features


# ==================== EXERCISES ====================
# 1. Try both RandomForest and LogisticRegression
# 2. Experiment with class_weight='balanced'
# 3. Add SMOTE (if you want) and compare