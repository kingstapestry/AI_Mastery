import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


"""
LESSON 24: Systematic Model Selection & Experiment Tracking

Goal:
    Learn how to professionally compare multiple models and choose the best one.
    This is a critical skill for real AI Engineering work.
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


# ==================== 3. CREATE MULTIPLE PIPELINES ====================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

pipelines = {
    "RandomForest_Default": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    "RandomForest_Balanced": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ]),
    "LogisticRegression": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=500))
    ])
}


# ==================== 4. SYSTEMATIC EVALUATION ====================
results = []

for name, model in pipelines.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, pred_proba),
        "Avg CV ROC-AUC": cv_scores.mean()
    })

    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, pred_proba):.4f}")
    print(f"Avg CV ROC-AUC: {cv_scores.mean():.4f}")


# ==================== 5. MODEL COMPARISON TABLE ====================
comparison_df = pd.DataFrame(results).round(4).sort_values(by="ROC-AUC", ascending=False)
print("\n=== Final Model Comparison ===")
print(comparison_df)


# ==================== 6. FEATURE IMPORTANCE FROM BEST MODEL ====================
print("\n=== Feature Importance from Best Model ===")
best_model_name = comparison_df.iloc[0]['Model']
best_pipeline = pipelines[best_model_name]
rf_classifier = best_pipeline.named_steps['classifier']

feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()

importances = rf_classifier.feature_importances_ if hasattr(rf_classifier, 'feature_importances_') else None

if importances is not None:
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    print(feature_importance.head(15))