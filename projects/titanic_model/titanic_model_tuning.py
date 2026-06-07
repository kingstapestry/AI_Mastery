import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


"""
Model Tuning (Hyperparameter Optimization)

Goal:
    Learn how to improve model performance by tuning hyperparameters.

Key Skills Learned:
    - Manual Tuning (changing n_estimators, max_depth, min_samples_split manually)
    - Automated Tuning with GridSearchCV
    - Understanding important RandomForest parameters
    - Comparing Base Model vs Manually Tuned vs GridSearch Tuned models
    - Using best_estimator_ and best_params_

Important Realization:
    Tuning doesn't always give big improvements on small/clean datasets.
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

# Convert categorical to numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

print("Data after cleaning:")
print(df.head())


# ==================== 3. FEATURE ENGINEERING ====================
print("\n--- Feature Engineering ---")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)


# ==================== 4. PREPARE DATA FOR MODEL ====================
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ==================== 5. BASE MODEL ====================
print("\n--- Base Model (Default Parameters) ---")

rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
base_pred = rf_base.predict(X_test)

print(f"Base Model Accuracy: {accuracy_score(y_test, base_pred):.4f}")


# ==================== 6. MANUAL TUNING ====================
print("\n--- Manual Tuning ---")

"""
Manual Tuning: Trying specific values by hand

n_estimators=200   → More decision trees (more stable)
max_depth=5        → Limit tree depth to reduce overfitting
min_samples_split=5 → Require more samples before splitting
"""

rf_manual = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
rf_manual.fit(X_train, y_train)
manual_pred = rf_manual.predict(X_test)

print(f"Manual Tuned Accuracy: {accuracy_score(y_test, manual_pred):.4f}")


# ==================== 7. ADVANCED TUNING - GRIDSEARCHCV ====================
print("\n--- GridSearchCV (Automated Tuning) ---")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")


# ==================== 8. FINAL EVALUATION ====================
print("\n--- Final Model Evaluation ---")

best_model = grid_search.best_estimator_
final_pred = best_model.predict(X_test)

print(f"Final Tuned Model Test Accuracy: {accuracy_score(y_test, final_pred):.4f}")


# ==================== SUMMARY ====================
"""
Key Takeaways from Model Tuning:

- Default parameters often work very well
- GridSearchCV automates the search for optimal hyperparameters
- Tuning doesn't always guarantee big improvements (especially on small datasets)
- Understanding parameters like n_estimators, max_depth helps build intuition
"""