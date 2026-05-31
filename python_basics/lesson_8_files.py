import os

# Writing to a file
with open("python_basics/my_first_note.txt", "w") as file:
    file.write("Hello! This is my AI Mastery journey.\n")
    file.write("Today I learned file handling.\n")

print("File written successfully!")


# Reading from a file
with open("python_basics/my_first_note.txt", "r") as file:
    content = file.read()
    print("\nFile Content:")
    print(content)


# Appending to a file
with open("python_basics/my_first_note.txt", "a") as file:
    file.write("I will become an AI expert!\n")

print("Content appended!")


# ==================== EXERCISES ====================
# Exercise 8.1:
# Write a function save_skills(skills_list) that takes a list of skills
# and saves each skill on a new line in a file called "my_skills.txt"

# Exercise 8.2:
# Write a function load_skills() that reads "my_skills.txt" and returns
# the skills as a list (remove the \n at the end of each line)

# Exercise 8.3:
# Call both functions: save your current skills, then load and print them.

# Write your code for Exercises 8.1 - 8.3 here:

def save_skills(skills_list: list):
    # Ensure directory exists
    os.makedirs("python_basics", exist_ok=True)

    with open("python_basics/my_skills.txt", "w") as file:
        for skill in skills_list:
            file.write(f"{skill}\n")

def load_skills():
    with open("python_basics/my_skills.txt", "r") as file:
        # Read lines and strip whitespace and \n
        list_of_contents = [contents.strip() for contents in file]
    print(list_of_contents)

skill_list = ["Python", "AI Mastery", "Automation"]

save_skills(skill_list)
load_skills()