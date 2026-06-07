import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


"""
Pipelines + Multi-Class Classification (Iris Dataset)

Goal:
    Learn professional ML practices using Scikit-Learn Pipelines and multi-class classification.

Key Skills Learned:
    - Working with multi-class problems (3 flower species)
    - Creating Pipelines (combining scaling + model in one object)
    - Benefits of Pipeline: cleaner code, prevents data leakage
    - Accessing steps inside a pipeline (named_steps)
    - Using GridSearchCV with Pipelines
    - Comparing models on an easy dataset (Iris)

Observation:
    On very simple datasets like Iris, even basic models can achieve very high accuracy.
"""


# ==================== 1. LOAD DATASET ====================
"""
The Iris dataset is a classic multi-class classification problem.
It has 3 classes of flowers: Setosa, Versicolor, Virginica
"""

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Feature Names:", iris.feature_names)
print("Target Names :", iris.target_names)
print("Dataset Shape:", X.shape)


# ==================== 2. TRAIN-TEST SPLIT ====================
# TODO: Split the data (test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================== 3. CREATE PIPELINE ====================
"""
Pipeline combines multiple steps into one object.
Benefits:
- Cleaner code
- Prevents data leakage
- Easier to use with GridSearchCV later
"""

# TODO: Create a Pipeline that does:
# 1. StandardScaler
# 2. RandomForestClassifier (or LogisticRegression)

pipeline_steps = [
    ('scaler', StandardScaler()),       # Step 1: Scale features
    ('classifier', RandomForestClassifier(random_state=42))     # Step 2: Model
]

model_pipeline = Pipeline(pipeline_steps)


# ==================== 4. TRAIN AND EVALUATE ====================
# TODO:
# 1. Fit the pipeline on training data
# 2. Make predictions on test data
# 3. Print accuracy, classification report, and confusion matrix

model_pipeline.fit(X_train, y_train)

pred = model_pipeline.predict(X_test)

print("\n=== Random Forest Model Evaluation ===")

print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))


# ==================== 5. TRY DIFFERENT MODELS ====================
# TODO: Create another pipeline using LogisticRegression and compare results

lr_pipeline_steps = [
    ('classifier', LogisticRegression())
]

lr_pipeline = Pipeline([('classifier', LogisticRegression(max_iter=500))])

lr_pipeline.fit(X_train, y_train)

lr_pred = lr_pipeline.predict(X_test)

print("\n=== Logistic Regression Evaluation ===")

print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")


# ==================== 6. FEATURE IMPORTANCE (Random Forest) ====================
# TODO: Access the RandomForest step inside the pipeline and show feature importance

print("\n=== Feature Importance (Random Forest) ===")

rf_model = model_pipeline.named_steps['classifier']

importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'Features': iris.feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feature_importance)


# ==================== 7. ADVANCED TUNING - GRIDSEARCHCV ====================
print("\n--- GridSearchCV (Automated Tuning) ---")

param_grid = {
    'classifier__n_estimators': [100, 200],     # Note the 'classifier__' prefix
    'classifier__max_depth': [None, 5, 10],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__min_samples_split': [2, 5]
}

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Test the best model
best_pred = grid_search.predict(X_test)
print(f"Best Model Test Accuracy: {accuracy_score(y_test, best_pred):.4f}")


# ==================== SUMMARY ====================
"""
Key Takeaways from Pipelines:
- Pipelines make your code cleaner and more maintainable
- They prevent data leakage (e.g. scaling before splitting)
- Very useful when combining with GridSearchCV
"""