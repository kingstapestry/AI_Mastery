import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""
Machine Learning Model to Predict: 

Did the passenger survive the Titanic disaster or not?
"""

# ==================== LOAD AND PREPARE DATA ====================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# TODO: Drop unnecessary columns (PassengerId, Name, Ticket, Cabin)
# TODO: Fill missing values in Age and Embarked
# TODO: Convert Sex and Embarked to numeric using .map()
"""
Removes columns that are not useful for prediction (ID, name, etc.).
Methods: 
    drop()
"""
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

"""
Fills missing values (NaN) with median for Age.
Methods: 
    fillna(df.median())

Fills missing values (NaN) with most common value for Embarked.
Methods: 
    fillna(df.mode()[0]) 
Notes: 
    mode() - Calculate most common value in that column
    [0] - Extracts very first item from that resulting series
"""
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

"""
Converts text (male/female, S/C/Q) into numbers because models only understand numbers.
Methods: 
    map({})
"""
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

"""Good to check after cleaning data."""
print("\nAfter Cleaning:\n", df.head())
print("\nData Types:\n", df.dtypes)

# TODO: Feature Engineering
# Create FamilySize and IsAlone columns
"""
Creates new useful information from existing columns.
Methods: 
    np.where()
Notes:
    where "FamilySize" == 1, "IsAlone" = 1, else "IsAlone" = 0
"""
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)

# TODO: Prepare features (X) and target (y)
"""
Prepare for Model
Methods:
    drop() 
Notes:
    X = Everything the model uses to make predictions
    y = What we want the model to predict (Survived or not)
"""
X = df.drop("Survived", axis=1)     # (inputs) everything except "Survived" to train model
y = df["Survived"]      # (target) what we want to predict

# TODO: Train-test split
"""
Train-Test Split
Functions:
    train_test_split(X, y, test_size, random_state)
Notes:
    Splits Data into:
        - Training set (model learns from this)
        - Test set (we check how good the model is on unseen data)

    test_size specifies what portion of your overall dataset should be allocated to the testing set (used to evaluate how well your model learns). The remainder is used for training:
        - Value Format: It can be represented as a decimal (e.g., 0.2 for 20%) or as an exact integer (e.g., 50 for 50 data points)
        - Standard Practice: A common split is 80% training and 20% testing (test_size = 0.2)
        - Why it matters: If your test size is too small, your evaluation might not be reliable. If it's too large, your machine learning model won't have enough data to learn effectively
    
    random_state before data is split, it is typically shuffled to prevent any order bias. random_state acts as a starting "seed" for the random number generator:
        - Before data is split, it is typically shuffled to prevent any order bias. random_state acts as a starting "seed" for the random number generator.
        - Without a seed: If you leave this parameter blank (None), your code will pull a completely different set of data into your training and testing groups every time you execute it, 
                            making your results impossible to debug or accurately compare
"""
# Use: train_test_split(..., test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================== MODEL TRAINING & EVALUATION ====================
# TODO: Create and train a RandomForestClassifier
# Use: RandomForestClassifier(n_estimators=100, random_state=42)
"""
Model Training
Classes:
    RandomForestClassifier(n_estimators, random_state)
Methods:
    fit(X_train, y_train)
Notes:
    n_estimators defines the total number of decision trees to include in the ensemble:
        - The Mechanism: Each tree makes its own isolated prediction based on its subset of data. The classifier then aggregates these votes to pick the overall majority outcome
        - Performance Impact: Generally, higher numbers of trees provide better accuracy and make predictions more stable. More trees also drastically minimize the risk of overfitting
        - The Limit: The benefits plateau after reaching an optimal threshold. 
                        Adding more trees past that point will significantly increase computation time and memory usage without giving you any meaningful performance boosts.
    
    random_state basically a RNG to make sure data is split

    .fit() method executes the actual training process of your Random Forest model by constructing the ensemble of decision trees based on your training data:
        - When you call this method, you pass your features (X) and target labels (y), triggering several backend operations

        - Data Sampling: The algorithm takes your dataset and creates n_estimators unique subsets using a process called bootstrap sampling (sampling with replacement)
        - Feature Selection: At every node of every individual tree, a random subset of features is selected to evaluate the best possible data split
        - Tree Building: The model builds out each decision tree fully based on those random combinations of data rows and features
        - Knowledge Storage: The algorithm saves all the resulting tree structures, split rules, and weights internally so it can make future predictions

        - KEY DETAIL - State Change: This method modifies the internal state of your model instance, changing it from an empty blueprint into a fully trained classifier
        - KEY DETAIL - Prerequisite: You must always call .fit() before you can use subsequent methods like .predict() to evaluate new data or .score() to check accuracy
"""
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

"""
Prediction & Basic Evaluation
Methods:
    predict(X_test)
    accuracy_score(y_test, rf_predictions)
    classification_report(y_test, rf_predictions)
    confusion_matrix(y_test, rf_predictions)
Notes:
    - Makes predictions on test data
    - Checks how accurate those predictions are
    - Confusion Matrix shows where the model is making mistakes
"""
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
"""
Cross Validation
Functions:
    cross_val_score(rf_model, X, y, cv=5)
Notes:
    - A more reliable way to check model performance by testing it on 5 different splits of the data
    - Instead of training your model just once on a single training set, it splits your data into five equal parts to test the model's reliability five separate times

    - Splits the Data: It divides your features (X) and targets (y) into 5 equal subsets (folds)
    - Isolates a Test Fold: It selects 1 fold to act as the validation set and groups the remaining 4 folds together to act as the training set
    - Fits the Model: It automatically calls .fit() on the 4 training folds using a fresh, un-trained copy of your model
    - Scores the Model: It evaluates the newly trained model on the isolated test fold and records the performance metric (default is accuracy for classifiers)
    - Repeats: It resets and repeats this entire process 5 times, ensuring every single data point gets used as part of a test set exactly once

    Iteration 1: [ TEST ] [ Train ] [ Train ] [ Train ] [ Train ]  --> Score #1
    Iteration 2: [ Train ] [ TEST ] [ Train ] [ Train ] [ Train ]  --> Score #2
    Iteration 3: [ Train ] [ Train ] [ TEST ] [ Train ] [ Train ]  --> Score #3
    Iteration 4: [ Train ] [ Train ] [ Train ] [ TEST ] [ Train ]  --> Score #4
    Iteration 5: [ Train ] [ Train ] [ Train ] [ Train ] [ TEST ]  --> Score #5
"""
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("\nCross Validation Scores:\n", cv_scores)
print("\nAverage Score:\n", cv_scores.mean())

# TODO: Show Feature Importance using model.feature_importances_
"""
Feature Importance
Notes:
    - Shows which features (columns) were most important for the models decisions

    rf_model.feature_importances_: This extracts a raw array of numerical scores from your trained model. Every score is between 0.0 and 1.0, and they all add up to exactly 1.0 (or 100%)

    pd.DataFrame(...): This maps those raw numerical scores directly to their corresponding column names (X.columns) so you can easily read which score belongs to which feature

    sort_values("Importance", ascending=False): This sorts the rows so the most influential features appear at the very top of your table, and the least important ones drop to the bottom
"""
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