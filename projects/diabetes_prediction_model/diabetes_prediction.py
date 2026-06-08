import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


"""
LESSON 19: Diabetes Prediction (Real-World Medical Dataset)

Goal:
    Work with a realistic, messy medical dataset to predict diabetes.
    Practice proper data cleaning and professional ML pipelines.

Key Skills:
    - Handling real-world data issues (0s representing missing values)
    - Medical data preprocessing
    - Using ROC-AUC for imbalanced classification
    - Building robust Pipelines
"""

# ==================== 1. LOAD DATA ====================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=column_names)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nTarget Distribution:\n", df['Outcome'].value_counts())


# ==================== 2. DATA EXPLORATION & CLEANING ====================
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


# ==================== 3. PREPARE FEATURES AND TARGET ====================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================== 4. CREATE PIPELINE ====================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])


# ==================== 5. TRAIN & EVALUATE ====================
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("\n=== Random Forest Pipeline Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))


# ==================== 6. ADVANCED EVALUATION ====================
# Use predicted probabilities for better ROC-AUC
pred_proba = pipeline.predict_proba(X_test)[:, 1]
print(f"\nROC AUC Score: {roc_auc_score(y_test, pred_proba):.4f}")


# ==================== 7. FEATURE IMPORTANCE ====================
rf_model = pipeline.named_steps['classifier']
importances = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)


# ==================== SUMMARY ====================
"""
Key Takeaways:
- In real medical data, 0 often means missing value, not actual zero
- Always inspect data carefully before modeling
- ROC-AUC is better than accuracy for imbalanced datasets
- Pipelines keep our workflow clean and professional
"""