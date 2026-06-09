import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score


"""
LESSON 22: Customer Churn Prediction (End-to-End Industry Project)

Goal:
    Build a realistic business project using mixed numeric + categorical data.
    Learn ColumnTransformer - a key tool for real-world tabular data.
"""

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


# ==================== 3. SEPARATE FEATURES ====================
numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in df.columns if col not in numeric_features + ["Churn"]]

print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)


# ==================== 4. CREATE PIPELINE WITH COLUMNTRANSFORMER ====================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

churn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# ==================== 5. TRAIN AND EVALUATE ====================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

churn_pipeline.fit(X_train, y_train)
churn_pred = churn_pipeline.predict(X_test)
churn_proba = churn_pipeline.predict_proba(X_test)[:, 1]

print("\n=== Model Performance ===")
print(f"Accuracy : {accuracy_score(y_test, churn_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, churn_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, churn_proba):.4f}")


# ==================== 6. FEATURE IMPORTANCE ====================
print("\n=== Feature Importance ===")

# Get feature names after transformation
feature_names = churn_pipeline.named_steps['preprocessor'].get_feature_names_out()

rf_model = churn_pipeline.named_steps['classifier']
importances = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(feature_importance.head(15))


# ==================== SUMMARY ====================
"""
Key Takeaways:
- ColumnTransformer is essential when you have mixed data types
- Always get feature names from the fitted preprocessor after OneHotEncoding
- Real business datasets are messy and require careful cleaning
"""