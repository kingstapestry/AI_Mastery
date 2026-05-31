import pandas as pd
import numpy as np
import random

print("=== AI Student Performance Tracker ===\n")

# ==================== YOUR CODE STARTS HERE ====================

# Task 1: Create a DataFrame with at least 8 students
# Columns: Name, Python_Score, Math_Score, Motivation, Hours_Studied
data = {
    "Name": ["Leighton", "Kramer", "Amelia", "Antonio", "Titus", "Nyla", "Maxton", "Jaylen"],
    # If random ints needed [random.randint(int_range_start, int_range_end) for _ in range(num_of_ints_needed)]
    "Python_Score": [73, 38, 80, 52, 54, 37, 65, 65],
    "Math_Score": [35, 96, 65, 43, 38, 37, 71, 34],
    "Motivation": [10, 9, 7, 4, 6, 8, 6, 8],
    "Hours_Studied": [180, 186, 149, 145, 180, 141, 148, 192]
}

df = pd.DataFrame(data)
print(df)

# Task 2: Add a column "Total_Score" = (Python_Score + Math_Score) / 2
df.insert(5, "Total_Score", (df["Python_Score"] + df["Math_Score"]) / 2)
print(df)

# Task 3: Add a column "Performance":
#    - "Excellent" if Total_Score >= 85
#    - "Good" if Total_Score >= 70
#    - "Needs Improvement" otherwise
#    (Hint: use .apply() or np.where)
df["Peformance"] = df["Total_Score"].apply(
    lambda score: "Excellent" if score >= 85 else("Good" if score >= 70 else "Needs Improvement")
)
print(df)

# Task 4: Save the final DataFrame to "ai_performance.csv"
df.to_csv("python_basics/ai_performance.csv", index=True)

# Task 5: Print the following:
#    - Top 3 students by Total_Score (sorted)
#    - Average Hours_Studied across all students
#    - Count of students in each Performance category
top_3_students = df.sort_values(by="Total_Score", ascending=False).head(3)
print(top_3_students)

average_hours = df["Hours_Studied"].mean()
print(f"Average: {average_hours:.2f} hours.")

perf_counts = df["Peformance"].value_counts()
print(perf_counts)

# ==================== YOUR CODE ENDS HERE ====================