import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


"""
TITANIC MODEL DEEP DIVE

Goal:
    Deepen understanding of model training, evaluation techniques, 
    and how to analyze model behavior beyond simple accuracy.

Key Concepts Covered:
    - Cross Validation (more reliable than single train/test split)
    - Confusion Matrix (shows types of prediction errors)
    - Feature Importance (which features influence decisions most)
    - Model comparison (Random Forest vs Logistic Regression)
"""

# ==================== 1. LOAD AND PREPARE DATA ====================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Original Dataset Shape:", df.shape)


# ==================== 2. DATA CLEANING ====================
print("\n--- Data Cleaning ---")

df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Convert categorical features to numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

print("Data after cleaning:")
print(df.head())
print("\nData Types:\n", df.dtypes)


# ==================== 3. FEATURE ENGINEERING ====================
print("\n--- Feature Engineering ---")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)

print("Features after engineering:", df.columns.tolist())


# ==================== 4. PREPARE DATA FOR MODEL ====================
X = df.drop("Survived", axis=1)     # Features
y = df["Survived"]                  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ==================== 5. MODEL TRAINING ====================
print("\n--- Training Random Forest ---")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)


# ==================== 6. MODEL EVALUATION ====================
print("\n=== Model Evaluation ===")

print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))


# ==================== 7. ADVANCED EVALUATION ====================
"""
Iteration 1: [ TEST ] [ Train ] [ Train ] [ Train ] [ Train ]  --> Score #1
Iteration 2: [ Train ] [ TEST ] [ Train ] [ Train ] [ Train ]  --> Score #2
Iteration 3: [ Train ] [ Train ] [ TEST ] [ Train ] [ Train ]  --> Score #3
Iteration 4: [ Train ] [ Train ] [ Train ] [ TEST ] [ Train ]  --> Score #4
Iteration 5: [ Train ] [ Train ] [ Train ] [ Train ] [ TEST ]  --> Score #5
"""
print("\n=== Advanced Evaluation ===")

# Cross Validation - More reliable performance estimate
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross Validation Scores:", cv_scores)
print(f"Average CV Score: {cv_scores.mean():.4f}")


# Feature Importance
print("\n=== Feature Importance ===")
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(feature_importance)


# ==================== 8. BONUS: COMPARE WITH LOGISTIC REGRESSION ====================
print("\n=== Logistic Regression Comparison ===")

lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_predictions):.4f}")


# ==================== SUMMARY ====================
"""
Key Takeaways from this Deep Dive:
- Cross Validation gives a more trustworthy estimate of model performance
- Confusion Matrix helps identify what kind of mistakes the model makes
- Feature Importance reveals which variables drive predictions
- Random Forest consistently outperforms Logistic Regression on this dataset
"""