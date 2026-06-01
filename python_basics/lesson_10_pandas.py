import pandas as pd
# Creating a DataFrame
data = {
    "Name": ["King", "Alex", "Sara", "Jordan", "Maria"],
    "Age": [25, 30, 22, 28, 24],
    "City": ["Vancouver", "Toronto", "Montreal", "Calgary", "Ottawa"],
    "AI_Score": [85, 72, 90, 65, 88],
    "Hours_Studied": [25, 18, 30, 15, 22]
}

df = pd.DataFrame(data)
print("Full DataFrame:")
print(df)

print("\nBasic Info:")
print(df.info())

print("\nStatistics:")
print(df.describe())


# Selecting data
print("\nAI Scores only:")
print(df["AI_Score"])

print("\nStudents with AI_Score > 80:")
print(df[df["AI_Score"] > 80])


# ==================== EXERCISES ====================
# Exercise 10.1:
# Add a new column called "Motivation_Level" with these values: [8, 7, 9, 6, 8]

# Exercise 10.2:
# Save the DataFrame to a CSV file called "ai_students.csv" in the root folder

# Exercise 10.3:
# Load the CSV file back into a variable called df2 and print df2

# Exercise 10.4:
# Filter and print students who are older than 24 AND have AI_Score > 75

# Exercise 10.5 (Challenge):
# Sort the DataFrame by AI_Score in descending order and print the top 3 students

# Write your solutions here:

df.insert(5, "Motivation_Level", [8, 7, 9, 6, 8])
df.to_csv("python_basics/ai_students.csv", index=True)      # Creates and saves to CSV

df2 = pd.read_csv("python_basics/ai_students.csv")      # Reads csv
print(df2)

print("\nStudents older than 24 and have AI Score > 75:")
print(df2[(df2["Age"] > 24) & (df2["AI_Score"] > 75)])      # Filter ages over 24 and ai score over 75

filtered_df = df2.sort_values(by="AI_Score", ascending=False)       # Sorting AI score values in descending order
print(filtered_df.head(3))      # Printing top 3 