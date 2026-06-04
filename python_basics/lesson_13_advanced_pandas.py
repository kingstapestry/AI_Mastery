import pandas as pd
import numpy as np

# Load the file we created earlier (if it exists)
# df = pd.read_csv("ai_performance.csv")

# Let's create fresh data for practice
np.random.seed(42)
data = {
    "Student_ID": range(1, 21),
    "Python_Score": np.random.randint(60, 100, 20),
    "Math_Score": np.random.randint(50, 95, 20),
    "Hours_Studied": np.random.randint(10, 50, 20),
    "Age": np.random.randint(18, 35, 20)
}

df = pd.DataFrame(data)

# Advanced operations
df["Total_Score"] = (df["Python_Score"] + df["Math_Score"]) / 2
df["Performance"] = pd.cut(df["Total_Score"], 
                          bins=[0, 70, 85, 100], 
                          labels=["Needs Work", "Good", "Excellent"])

print("Shape:", df.shape)
print("\nTop 5 students:")
print(df.nlargest(5, "Total_Score"))


# ==================== EXERCISES ====================
# Exercise 13.1:
# Add a new column "Study_Efficiency" = Total_Score / Hours_Studied

# Exercise 13.2:
# Find the student with highest Study_Efficiency

# Exercise 13.3:
# Group by Performance and show mean Python_Score and Hours_Studied for each group

# Exercise 13.4 (Harder):
# Create a new column "Age_Group" using pd.cut():
#   18-22 → "Young"
#   23-28 → "Mid"
#   29+   → "Senior"

# Write your code for all exercises here:

df["Study_Efficiency"] = (df["Total_Score"] / df["Hours_Studied"])
print(f"\nStudent with highest study efficiency:\n{df.nlargest(1, "Study_Efficiency")}")

mean_scores = df.groupby("Performance")[["Python_Score", "Hours_Studied"]].mean()
print(f"\nGrouped by Performance:\n{mean_scores}")

df["Age_Group"] = pd.cut(df["Age"],
                         bins=[17, 22, 28, np.inf],
                         labels=["Young", "Mid", "Senior"])

print(df.head())