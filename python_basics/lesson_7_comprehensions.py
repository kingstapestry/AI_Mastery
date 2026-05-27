numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# List comprehensions (very powerful in AI)
squares = [x**2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]

print("Squares:", squares)
print("Even numbers:", evens)

odd_squares = [x**2 for x in numbers if x % 2 == 1]
print("Odd squares:", odd_squares)


# ==================== EXERCISES ====================
# Exercise 7.1:
# Create a list called ai_scores with 10 numbers between 1 and 100 (you can hardcode)

# Exercise 7.2:
# Use list comprehension to create a new list called passed_scores
# that only keeps scores >= 70

# Exercise 7.3:
# Print how many people passed (use len())

# Write your code for Exercises 7.1 - 7.3 here:

ai_scores = [1, 5, 15, 2, 23, 50, 60, 100, 95, 75]
passed_scores = [x for x in ai_scores if x >= 70]

print(f"The number of people who passed is: {len(passed_scores)}")