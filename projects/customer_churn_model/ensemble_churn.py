import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


"""
LESSON 23: Model Ensembling & Stacking

Goal:
    Learn how to combine multiple models to create stronger predictions.
"""

# ==================== 1. LOAD & CLEAN DATA ====================
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

df = df.drop(["customerID"], axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("Data cleaned successfully.")


# ==================== 2. PREPARE DATA ====================
numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in df.columns if col not in numeric_features + ["Churn"]]

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==================== 3. CREATE PREPROCESSOR ====================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)


# ==================== 4. CREATE BASE MODELS ====================
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=500))
])


# ==================== 5. VOTING CLASSIFIER ====================
print("\n--- Voting Classifier ---")

voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('lr', lr_model)],
    voting='soft'          # 'soft' uses probabilities, usually better
)

voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
voting_proba = voting_clf.predict_proba(X_test)[:, 1]

print(f"Voting Classifier - Accuracy: {accuracy_score(y_test, voting_pred):.4f}")
print(f"Voting Classifier - F1 Score: {f1_score(y_test, voting_pred):.4f}")
print(f"Voting Classifier - ROC-AUC : {roc_auc_score(y_test, voting_proba):.4f}")


# ==================== 6. STACKING CLASSIFIER ====================
print("\n--- Stacking Classifier ---")

stacking_clf = StackingClassifier(
    estimators=[('rf', rf_model), ('lr', lr_model)],
    final_estimator=LogisticRegression(max_iter=500),
    cv=5,
    passthrough=False
)

stacking_clf.fit(X_train, y_train)
stack_pred = stacking_clf.predict(X_test)
stack_proba = stacking_clf.predict_proba(X_test)[:, 1]

print(f"Stacking Classifier - Accuracy: {accuracy_score(y_test, stack_pred):.4f}")
print(f"Stacking Classifier - F1 Score: {f1_score(y_test, stack_pred):.4f}")
print(f"Stacking Classifier - ROC-AUC : {roc_auc_score(y_test, stack_proba):.4f}")


 # ==================== 7. FEATURE IMPORTANCE ====================
print("\n=== Feature Importance from Stacking Classifier ===")

# Access the Random Forest base estimator inside the stacking model
rf_base_model = stacking_clf.named_estimators_['rf']        # 'rf' is the name we gave it
rf_classifier = rf_base_model.named_steps['classifier']     # Access the actual RandomForest

importances = rf_classifier.feature_importances_

# Get the final feature names after ColumnTransformer
feature_names = stacking_clf.named_estimators_['rf'].named_steps['preprocessor'].get_feature_names_out()

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)
print(feature_importance.head(15))

# # Best and simplest way:
# rf_model = rf_pipeline  # or your original Random Forest pipeline

# rf_classifier = rf_model.named_steps['classifier']
# feature_names = rf_model.named_steps['preprocessor'].get_feature_names_out()

# # Then create the DataFrame as usual


# ==================== SUMMARY ====================
"""
Key Takeaways:
- Ensembling (Voting & Stacking) can often improve performance
- Stacking is more powerful but slower
- Voting is simpler and faster
"""