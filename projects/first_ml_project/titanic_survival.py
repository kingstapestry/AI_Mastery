import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

"""
Machine Learning Model to Predict: 

Did the passenger survive the Titanic disaster or not?
"""

# ==================== LOAD DATA ====================
# We'll use Titanic dataset (classic beginner ML project)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print(df.head())

# ==================== DATA CLEANING ====================
# Drop columns that won't be useful: PassengerId, Name, Ticket, Cabin
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Convert Sex to numeric (0 = female, 1 = male)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})       # IMPORTANT: scikit-learn models cannot handle strings - they only need numbers

print("\nAfter Cleaning:\n", df.head())
print("\nData Types:\n", df.dtypes)     # good to check after cleaning data

# ==================== FEATURE ENGINEERING ====================
# Create new feature FamilySize = SibSp + Parch + 1
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Create one more feature: "IsAlone" = 1 if FamilySize == 1 else 0
# df["IsAlone"] = df["FamilySize"].apply(lambda x: 1 if x == 1 else 0)
df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)

print("\nAfter Feature Engineering:\n", df.head())

# ==================== PREPARE FOR MODEL ====================
# Select features and target (Survived)
"""
X = All columns except Survived. These are the features (information) the model will learn from.
y = Only the Survived column. This is the correct answer the model tries to learn.
"""
X = df.drop("Survived", axis=1)     # Features (inputs)
y = df["Survived"]                  # Target (what we want to predict)

# Split data into train and test (80/20)
"""
Divide data into training (80%) and testing (20%).
The model learns from the training data (X_train, y_train).
We test how good it is on unseen data (X_test, y_test).

IMPORTANT: To check if the model is actually learning or just memorizing.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==================== TRAIN MODEL ====================
# Train a RandomForestClassifier
"""
RandomForestClassifier() → We create the model (think of it as a smart student).
.fit(X_train, y_train) → We train the model. We show it thousands of examples (features + correct answers), and it learns patterns.

For example, the model might learn:
"Women (Sex=1) had higher chance of survival"
"Rich people (high Fare) had higher chance"
"""
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)

# Make predictions on test set
"""
After training we test it: 

The model looks at X_test (new passengers it hasn't seen) and makes predictions.
Then we compare predictions with the real y_test to calculate Accuracy.
"""
rf_predictions = rf_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

# Print accuracy and classification report
print("RF Accuracy:", accuracy_score(y_test, rf_predictions))
print("\nRF Classification Report:\n", classification_report(y_test, rf_predictions))

print("LR Accuracy:", accuracy_score(y_test, lr_predictions))
print("\nLR Classfication Report:\n", classification_report(y_test, lr_predictions))

importances = rf_model.feature_importances_
feature_names = X.columns

for name, score in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(name, ":", round(score, 4))

joblib.dump(rf_model, "projects/first_ml_project/titanic_model.pkl")
print("Model saved!")

# ==================== EXERCISES (Do all of them) ====================
# Exercise 14.1: Complete the TODOs above to make the model run

# Exercise 14.2: Try different models (try LogisticRegression too) 
# and compare which one performs better

# Exercise 14.3: Add feature importance visualization (model.feature_importances_)

# Exercise 14.4 (Certification Style):
# What is the difference between train_test_split and cross-validation?
# Why do we use random_state=42?

# Exercise 14.5 (Challenge):
# Try creating one more feature: "IsAlone" = 1 if FamilySize == 1 else 0
# Does it improve accuracy?

# Bonus Challenge:
# Save the trained model using joblib