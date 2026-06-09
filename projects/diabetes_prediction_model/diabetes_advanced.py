import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


"""
LESSON 20: Handling Imbalanced Data (Diabetes Prediction - Advanced)

Goal:
    Learn how to deal with imbalanced datasets, which are very common in real-world problems 
    (fraud detection, medical diagnosis, churn prediction, etc.).
"""

# ==================== 1. LOAD & PREPARE DATA ====================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=column_names)

print("Dataset Shape:", df.shape)
print("\nOriginal Class Distribution:\n", df['Outcome'].value_counts())


# ==================== 2. DATA CLEANING ====================
print("\n--- Data Cleaning ---")

# Replace 0s with NaN in medically impossible columns
cols_with_zeros = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Fill with median (robust choice for medical data)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())

print("Data cleaned successfully.")


# ==================== 3. PREPARE FEATURES AND TARGET ====================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==================== 4. BASIC PIPELINE ====================
print("\n=== Basic Random Forest Pipeline ===")

basic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

basic_pipeline.fit(X_train, y_train)
basic_pred = basic_pipeline.predict(X_test)
basic_proba = basic_pipeline.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, basic_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, basic_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, basic_proba):.4f}")


# ==================== 5. ADVANCED PIPELINE WITH SMOTE ====================
print("\n=== SMOTE Pipeline ===")

smote_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

smote_pipeline.fit(X_train, y_train)
smote_pred = smote_pipeline.predict(X_test)
smote_proba = smote_pipeline.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, smote_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, smote_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, smote_proba):.4f}")


# ==================== 6. FEATURE IMPORTANCE ====================
print("\n=== Feature Importance (Basic Model) ===")
rf_model = basic_pipeline.named_steps['classifier']
importances = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(feature_importance)


# ==================== SUMMARY ====================
"""
Key Takeaway:
Even though SMOTE is popular, it doesn't always improve performance.
Always experiment with multiple techniques (class_weight, SMOTE, undersampling, etc.).
"""