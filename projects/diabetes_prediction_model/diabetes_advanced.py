import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ==================== 1. LOAD & PREPARE DATA ====================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=column_names)

# TODO: Clean the data (replace 0s with NaN and fill with median) - reuse from Lesson 19
print("\n--- Data Exploration & Cleaning ---")

print("Missing Values (0s in medical columns are often missing data):")
print(df.eq(0).sum())

# Replace 0s with NaN in medically impossible columns
cols_with_zeros = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Fill missing values with median (robust for medical data)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())

print("\nData after cleaning:")
print(df.head())

# ==================== 2. PREPARE FEATURES AND TARGET ====================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("Class Distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # Important for imbalanced data
)


# ==================== 3. BASIC PIPELINE (No Balancing) ====================
# TODO: Create a normal Pipeline with StandardScaler + RandomForestClassifier


# ==================== 4. ADVANCED PIPELINE WITH SMOTE ====================
"""
SMOTE = Synthetic Minority Over-sampling Technique
It creates synthetic examples of the minority class to balance the dataset.
"""

# TODO: Create an ImbPipeline that does:
# 1. StandardScaler
# 2. SMOTE
# 3. RandomForestClassifier


# ==================== 5. TRAIN & COMPARE BOTH PIPELINES ====================
# TODO: Train both pipelines and compare:
# - Accuracy
# - F1-Score
# - ROC-AUC


# ==================== 6. FEATURE IMPORTANCE & FINAL EVALUATION ====================
# TODO: Show feature importance from the best model


# ==================== EXERCISES ====================
# 1. Try class_weight='balanced' in RandomForest instead of SMOTE
# 2. Compare Logistic Regression with SMOTE
# 3. Try different values of k_neighbors in SMOTE