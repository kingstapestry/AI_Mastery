import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


"""
HOUSE PRICE PREDICTION - Regression Project

Goal:
    Predict continuous house prices (medv) using the Boston Housing dataset.

Key Learning Objectives:
    - Difference between Classification and Regression problems
    - Using RandomForestRegressor vs LinearRegression
    - Feature Scaling with StandardScaler
    - Regression evaluation metrics (MAE, RMSE, R²)
"""

# ==================== 1. LOAD DATA ====================
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())


# ==================== 2. DATA EXPLORATION & CLEANING ====================
print("\n--- Data Exploration & Cleaning ---")

print("Dataset Info:")
print(df.info())

print("\nMissing Values per Column:")
print(df.isnull().sum())

print("\nPercentage of Missing Values:")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\nBasic Statistics:")
print(df.describe())


# ==================== 3. PREPARE FEATURES AND TARGET ====================
X = df.drop("medv", axis=1)     # All features except target
y = df["medv"]                  # Target: Median house value

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")


# ==================== 4. FEATURE SCALING ====================
print("\n--- Feature Scaling ---")

"""
Why scale?
- Features have very different scales (e.g. number of rooms vs crime rate)
- Linear models are very sensitive to scale
- Tree-based models (RandomForest) are less affected, but scaling is still good practice
"""

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier reading
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Features successfully scaled!")


# ==================== 5. TRAIN MODELS ====================
print("\n--- Training Models ---")

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_model_scaled = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_scaled.fit(X_train_scaled, y_train)

# Linear Regression (Baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# ==================== 6. MAKE PREDICTIONS ====================
rf_pred = rf_model.predict(X_test)
rf_pred_scaled = rf_model_scaled.predict(X_test_scaled)
lr_pred = lr_model.predict(X_test)


# ==================== 7. MODEL EVALUATION ====================
print("\n=== MODEL PERFORMANCE COMPARISON ===\n")

print("Random Forest (No Scaling):")
print(f"  R² Score : {r2_score(y_test, rf_pred):.4f}")
print(f"  MAE      : ${mean_absolute_error(y_test, rf_pred):.2f}")
print(f"  RMSE     : ${mean_squared_error(y_test, rf_pred)**0.5:.2f}")

print("\nRandom Forest (With Scaling):")
print(f"  R² Score : {r2_score(y_test, rf_pred_scaled):.4f}")

print("\nLinear Regression:")
print(f"  R² Score : {r2_score(y_test, lr_pred):.4f}")
print(f"  MAE      : ${mean_absolute_error(y_test, lr_pred):.2f}")
print(f"  RMSE     : ${mean_squared_error(y_test, lr_pred)**0.5:.2f}")


# ==================== 8. FEATURE IMPORTANCE ====================
print("\n=== Feature Importance (Random Forest) ===")
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(feature_importance)


# ==================== SUMMARY ====================
"""
Key Takeaways from House Price Project:

- Regression problems predict continuous values (not categories)
- RandomForestRegressor significantly outperforms LinearRegression
- Feature Scaling is crucial for Linear Models but less important for Random Forest
- R² Score is the main evaluation metric for regression (closer to 1.0 = better)
- Always compare multiple models to understand what works best for your data
"""