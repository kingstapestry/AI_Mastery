import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==================== LOAD AND PREPARE DATA ====================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# TODO: Drop unnecessary columns (PassengerId, Name, Ticket, Cabin)
# TODO: Fill missing values in Age and Embarked
# TODO: Convert Sex and Embarked to numeric using .map()
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1})

# TODO: Feature Engineering
# Create FamilySize and IsAlone columns
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)

# TODO: Prepare features (X) and target (y)
X = df.drop("Survived", axis=1)     # (inputs) everything except "Survived" to train model
y = df["Survived"]      # (target) what we want to predict

# TODO: Train-test split
# Use: train_test_split(..., test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================== MODEL TRAINING & EVALUATION ====================
# TODO: Create and train a RandomForestClassifier
# Use: RandomForestClassifier(n_estimators=100, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# TODO: Make predictions on the test set
rf_predictions = rf_model.predict(X_test)

# TODO: Print Accuracy and Classification Report
print("\nAccuracy Score:\n", accuracy_score(y_test, rf_predictions))
print("\nClassification Report:\n", classification_report(y_test, rf_predictions))

# TODO: Print Confusion Matrix
cm = confusion_matrix(y_test, rf_predictions)
print("\nConfusion Matrix:\n", cm)

# ==================== ADVANCED EVALUATION ====================
# TODO: Perform 5-fold Cross Validation using cross_val_score
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("\nCross Validation Scores:\n", cv_scores)
print("\nAverage Score:\n", cv_scores.mean())

# TODO: Show Feature Importance using model.feature_importances_
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)
print(feature_importance)

# ==================== EXERCISES ====================
# 1. Try training a LogisticRegression model and compare accuracy
# 2. Experiment with different values of n_estimators and max_depth
# 3. Add comments explaining what each major step does