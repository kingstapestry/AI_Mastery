import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = {
    "Name": ["King", "Alex", "Sara", "Jordan", "Maria", "David", "Emma"],
    "AI_Score": [85, 72, 90, 65, 88, 95, 78],
    "Hours_Studied": [25, 18, 30, 15, 22, 35, 20],
    "Motivation": [8, 7, 9, 6, 8, 10, 7]
}

df = pd.DataFrame(data)

# ==================== MAIN FIGURE WITH 4 CHARTS ====================
# Create a big canvas (figure) that can hold multiple small charts
plt.figure(figsize=(12, 10))     # Makes the whole window 12 inches wide and 10 inches tall

# ------------------- Chart 1: Line Plot -------------------
plt.subplot(2, 2, 1)    # Means: 2 rows, 2 columns, this is position 1 (top-left)
                        # Think of it as dividing the big window into 4 small boxes

plt.plot(df["Name"], df["AI_Score"], 
         marker='o',         # Put a circle (dot) on each data point
         color='blue',       # Line color
         linewidth=2)        # Make the line a bit thicker

plt.title("AI Score by Student")   # Title at the top of this chart
plt.xticks(rotation=45)            # Rotate student names so they don't overlap
plt.ylabel("AI Score")             # Label for the up-down direction
plt.grid(True, alpha=0.3)          # Light grid lines to help read values


# ------------------- Chart 2: Bar Plot (Hours) -------------------
plt.subplot(2, 2, 2)    # Position 2 (top-right)

plt.bar(df["Name"], df["Hours_Studied"], color='green')  # Tall bars for each student
plt.title("Hours Studied")
plt.xticks(rotation=45)
plt.ylabel("Hours")
plt.grid(True, alpha=0.3)


# ------------------- Chart 3: Scatter Plot (Challenge) -------------------
# Create a scatter plot between Hours_Studied and AI_Score.
# Color the points based on Motivation level (use c=df["Motivation"])
plt.subplot(2, 2, 3)    # Position 3 (bottom-left)

# Scatter plot = dots on a graph
scatter = plt.scatter(
    df["Hours_Studied"],      # X-axis: Hours studied
    df["AI_Score"],           # Y-axis: AI Score
    c=df["Motivation"],       # Color the dots based on motivation level
    cmap="viridis",           # Use a nice color scale (dark to bright)
    s=100                     # Size of each dot
)

plt.title("Hours Studied vs AI Score\n(colored by Motivation)")
plt.xlabel("Hours Studied")   # Label for left-to-right
plt.ylabel("AI Score")
plt.colorbar(scatter, label="Motivation Level")  # Shows color meaning on the side
plt.grid(True, alpha=0.3)


# ------------------- Chart 4: Histogram (Exercise 12.1) -------------------
# Create a histogram of AI_Score (use plt.hist())
plt.subplot(2, 2, 4)    # Position 4 (bottom-right)

plt.hist(df["AI_Score"], 
         bins=5,              # Divide scores into 5 groups
         color='skyblue',     # Fill color
         edgecolor='black')   # Border around each bar

plt.title("Histogram of AI Scores")
plt.xlabel("AI Score")
plt.ylabel("Frequency")       # How many students in each score range
plt.grid(axis="y", alpha=0.75)  # Only horizontal lines


# Final touches for the big figure
plt.tight_layout()   # Automatically adjusts spacing so titles/labels don't overlap
plt.show()           # Display all 4 charts


# ==================== EXERCISE 12.2: Separate Motivation Bar Plot ====================
# Create a bar plot showing Motivation levels
plt.figure(figsize=(8, 6))   # New separate canvas

plt.bar(df["Name"], df["Motivation"], color='purple')
plt.title("Motivation Levels by Student")
plt.xticks(rotation=45)
plt.ylabel("Motivation Level (1-10)")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# ==================== NOTES ====================
# plt.figure() → Creates a blank paper to draw on
# plt.subplot() → Divides the paper into smaller sections
# plt.plot() → Draws connected lines
# plt.bar() → Draws rectangular bars
# plt.scatter() → Draws individual dots
# plt.hist() → Draws a histogram (shows distribution)
# plt.title(), plt.xlabel(), plt.ylabel() → Add labels
# plt.xticks(rotation=45) → Fixes overlapping names
# plt.grid() → Adds helpful grid lines
# plt.tight_layout() → Cleans up spacing