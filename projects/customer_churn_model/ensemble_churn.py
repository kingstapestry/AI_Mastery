import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ==================== 1. LOAD DATA ====================
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print("\nChurn Distribution:\n", df['Churn'].value_counts())


# ==================== 2. DATA CLEANING ====================
print("\n--- Data Cleaning ---")

df = df.drop(["customerID"], axis=1)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Convert target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("Data cleaned successfully.")


# ==================== 2. PREPARE DATA ====================
# TODO: Define numeric_features and categorical_features
# TODO: Train-test split with stratify=y


# ==================== 3. CREATE BASE MODELS ====================
# TODO: Create individual pipelines for:
# - RandomForestClassifier
# - LogisticRegression


# ==================== 4. VOTING CLASSIFIER (Simple Ensemble) ====================
# TODO: Create a VotingClassifier that combines RandomForest + LogisticRegression
# (hard voting and soft voting)


# ==================== 5. STACKING CLASSIFIER (Advanced Ensemble) ====================
# TODO: Create a StackingClassifier 
# - Base estimators: RandomForest + LogisticRegression
# - Final estimator: LogisticRegression


# ==================== 6. TRAIN & COMPARE ALL MODELS ====================
# TODO: Train all models (Base RF, Voting, Stacking) and compare:
# - Accuracy
# - F1 Score
# - ROC-AUC


# ==================== 7. FEATURE IMPORTANCE FROM BEST MODEL ====================
# TODO: Show feature importance from the best performing model


# ==================== SUMMARY ====================
# TODO: Write your conclusion on which ensemble method worked best