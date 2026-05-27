# Dictionary example
student = {
    "name": "King",
    "age": 25,
    "city": "Vancouver",
    "skills": ["Python", "AI"],
    "motivation": 10,
    "level": "beginner"
}

print(student["name"])
print(student.get("age"))

# Add new information
student["favorite_model"] = "GPT"
print(student)


# Loop through dictionary
print("\n--- Student Profile ---")
for key, value in student.items():
    print(key, ":", value)


# ==================== EXERCISES ====================
# Exercise 6.1:
# Create a dictionary called ai_project with these keys:
# - project_name
# - difficulty (number 1-10)
# - expected_days
# - technologies (a list of strings)

# Exercise 6.2:
# Write a function called print_project_info(project) that takes the dictionary
# and prints all information in a nice formatted way.

# Write your code for Exercise 6.1 and 6.2 here:

ai_project = {
    "project_name": "AI Mastery",
    "difficulty": 8,
    "expected_days": 90,
    "technologies": ["Python", "AI", "Many Other Things"]
}

def print_project_info(project):
    for key, value in ai_project.items():
        print(key, ":", value)

print_project_info(ai_project)