import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


"""
MODEL COMPARISON SUMMARY - LESSONS 14 to 17

1. RandomForestClassifier (Titanic - Lessons 14-16)
   - Type: Classification Model
   - Task: Predict discrete categories (Survived = 1 or 0)
   - Output: Class label (0 or 1)
   - Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
   - Used for: Binary or Multi-class classification problems

2. LogisticRegression (Titanic - Lesson 15)
   - Type: Linear Classification Model
   - Task: Also classification (binary)
   - Simpler and faster than Random Forest
   - Usually less accurate on complex datasets compared to Random Forest
   - Good as a baseline model

3. RandomForestRegressor (House Price Prediction - Lesson 17)
   - Type: Regression Model
   - Task: Predict continuous numerical values (house prices)
   - Output: A real number (e.g. 25.6 instead of 0 or 1)
   - Evaluation Metrics: MAE, RMSE, R² Score
   - Used for: Predicting quantities, prices, temperatures, etc.

Key Differences:

- Classifier → Predicts categories / labels (Yes/No, Spam/Not Spam, etc.)
- Regressor   → Predicts continuous numbers (prices, scores, sales, etc.)

RandomForest (both Classifier and Regressor):
- Ensemble method (uses many decision trees)
- Very powerful and robust
- Handles non-linear relationships well
- Less prone to overfitting compared to single decision tree

Why we tune models:
- Default parameters are okay, but tuning (Manual or GridSearchCV) can improve performance.
"""


# ==================== 1. LOAD AND PREPARE DATA ====================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# TODO: Data Cleaning
# 1. Drop these columns: PassengerId, Name, Ticket, Cabin
# 2. Fill missing Age with median value
# 3. Fill missing Embarked with the most common value
# 4. Convert Sex column: male → 0, female → 1
# 5. Convert Embarked column: S → 0, C → 1, Q → 2

df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map(lambda s: 0 if s == "male" else 1)
df["Embarked"] = df["Embarked"].map(lambda e: 0 if e == "S" else(1 if e == "C" else 2))

# print(df.head())
# print(df.dtypes)

# TODO: Feature Engineering
# 1. Create FamilySize = SibSp + Parch + 1
# 2. Create IsAlone = 1 if FamilySize == 1 else 0

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = df["FamilySize"].apply(lambda f: 1 if f == 1 else 0)

print(df.head())
print(df.dtypes)

# TODO: Prepare features and target
# X = all columns except Survived
# y = Survived column

X = df.drop("Survived", axis=1)
y = df["Survived"]

# TODO: Train-Test Split
# Use test_size=0.2 and random_state=42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================== 2. BASE MODEL ====================
# TODO: 
# 1. Create a RandomForestClassifier with n_estimators=100 and random_state=42
# 2. Train it using .fit(X_train, y_train)
# 3. Make predictions on X_test
# 4. Print the accuracy

rf_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_1.fit(X_train, y_train)

rf_model_1_predictions = rf_model_1.predict(X_test)

print("\nAccuracy Score (RF-1):", accuracy_score(y_test, rf_model_1_predictions))
print("\nClassification Report (RF-1):\n", classification_report(y_test, rf_model_1_predictions))


# ==================== 3. MANUAL TUNING ====================
# TODO:
# Create a new RandomForestClassifier with these parameters:
# n_estimators=200, max_depth=5, min_samples_split=5, random_state=42
# Train it, predict, and print accuracy

"""
MANUAL TUNING: Trying specific values by hand to see if we can beat default model

n_estimators=200: Number of decision trees in the forest. More trees = more stable predictions (but slower)
max_depth=5: Limits how deep each tree can grow. Prevents the model from becoming too complex and overfitting
min_samples_split=5: Minimum number of samples required to split a node. Helps control overfitting
random_state=42: Makes results reproducible
"""

rf_model_2 = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=5, random_state=42)
rf_model_2.fit(X_train, y_train)

rf_model_2_predictions = rf_model_2.predict(X_test)

print("\nAccuracy Score (RF-2):", accuracy_score(y_test, rf_model_2_predictions))
print("\nClassification Report (RF-2):\n", classification_report(y_test, rf_model_2_predictions))


# ==================== 4. ADVANCED TUNING WITH GRIDSEARCHCV ====================
# TODO: Define param_grid (copy this and modify if you want):

"""
GRIDSEARCHCV: Automatically searches through many combinations to find optimal settings

param_grid: A dictionary that defines all the combinations you want to try. GridSearch will test every possible combination
GridSearchCV: Automatically tries all combinations from param_grid and evaluates them using cross-validation
cv=5: 5-fold cross validation (more reliable than single train/test split)
n_jobs=-1: Uses all CPU cores to run faster

grid_search.best_params_: Shows which combination gave the best result
grid_search.best_score_: The best cross-validation score achieved
"""

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5]
}

# TODO:
# 1. Create GridSearchCV using RandomForestClassifier(random_state=42)
# 2. Run grid_search.fit(X_train, y_train)
# 3. Print the best parameters: grid_search.best_params_
# 4. Print the best score: grid_search.best_score_

rf_final = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_final, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"\nBest Score: {grid_search.best_score_:.4f}")


# ==================== 5. FINAL EVALUATION ====================
# TODO: Train final model using the best parameters and check accuracy on test set

"""
Final Evaluation

grid_search.best_estimator_: This gives you the best trained model (already fitted with the winning parameters)
.score(X_test, y_test): A quick way to get accuracy on the test set

We use this final model to check real-world performance on unseen data
"""

best_rf_model = grid_search.best_estimator_
test_accuracy = best_rf_model.score(X_test, y_test)

print(f"Test Set Accuracy (RF-Final): {test_accuracy:.4f}")


# ==================== EXERCISES ====================
# 1. Compare accuracy of Base Model vs Manually Tuned vs GridSearch Tuned
# 2. Try adding one more parameter to param_grid (example: max_features=[ 'sqrt', 'log2'])