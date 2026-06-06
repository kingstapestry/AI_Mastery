import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

"""
TITANIC SURVIVAL PREDICTION - First ML Project

Goal:
    Build a Machine Learning model to predict whether a passenger 
    survived the Titanic disaster (Binary Classification).

Key Learning Points:
    - Full ML Pipeline: Load → Clean → Feature Engineering → Train → Evaluate
    - Difference between Classifier and Regressor
    - Importance of converting categorical data to numeric
    - Feature Importance analysis
    - Saving trained models with joblib
"""

# ==================== 1. LOAD DATA ====================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())


# ==================== 2. DATA CLEANING ====================
print("\n--- Data Cleaning ---")

# Drop irrelevant columns
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Convert categorical features to numeric (ML models need numbers)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

print("Data after cleaning:")
print(df.head())
print("\nData Types:\n", df.dtypes)


# ==================== 3. FEATURE ENGINEERING ====================
print("\n--- Feature Engineering ---")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)

print("New features added. Current columns:", df.columns.tolist())


# ==================== 4. PREPARE DATA FOR MODEL ====================
X = df.drop("Survived", axis=1)     # Features (input)
y = df["Survived"]                  # Target (output)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ==================== 5. MODEL TRAINING ====================
print("\n--- Training Models ---")

# Random Forest (usually the stronger model)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Logistic Regression (simple baseline)
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)


# ==================== 6. PREDICTION & EVALUATION ====================
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print("\n=== Model Performance ===")
print(f"Random Forest Accuracy : {accuracy_score(y_test, rf_pred):.4f}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))


# ==================== 7. FEATURE IMPORTANCE ====================
print("\n=== Feature Importance (Random Forest) ===")
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feature_importance)


# ==================== 8. SAVE MODEL ====================
joblib.dump(rf_model, "projects/titanic_model/titanic_model.pkl")
print("\nModel saved as 'projects/titanic_model/titanic_model.pkl'")


# ==================== SUMMARY ====================
"""
Takeaways from this project:
- Always convert categorical data to numbers before training
- RandomForest generally outperforms LogisticRegression on tabular data
- Feature Engineering (FamilySize, IsAlone) can improve model performance
- Feature Importance helps understand what drives predictions
"""