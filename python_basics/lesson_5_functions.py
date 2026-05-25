# Basic function
def greet(name):
    print(f"Hello {name}! Welcome to AI Mastery.")

greet("King")


# Function with return value
def add_numbers(a, b):
    return a + b

result = add_numbers(15, 27)
print("Sum is:", result)


# Function with default parameters
def describe_person(name, age=25, city="Vancouver"):
    print(f"{name} is {age} years old from {city}.")


describe_person("King")
describe_person("Alex", 30, "Toronto")


# Function that takes a list
def print_skills(skills_list):
    print("My Current AI Skills:")
    for skill in skills_list:
        print("→", skill)


my_skills = ["Python", "VS Code", "Logic"]
print_skills(my_skills)


# ==================== EXERCISES ====================
# Exercise 5.1:
# Create a function called calculate_ai_score that takes 3 parameters:
# python_level, math_level, motivation (all 1-10)
# Return the average of the three numbers.
# Then print a message:
#   >=8  → "Future AI Master!"
#   >=6  → "Strong progress!"
#   <6   → "Keep grinding King!"

# Write your Exercise 5.1 code here:

def caculate_ai_score(python_level: int, math_level: int, motivation: int):
    average = (python_level + math_level + motivation) / 3

    if average >= 8:
        print("Future AI Master!")
    elif average >= 6:
        print("Strong progress!")
    elif average < 6:
        print("Keep grinding King!")

    return average

caculate_ai_score(5, 10, 10)


# Exercise 5.2:
# Create a function called print_welcome that takes name and level,
# and prints a motivational message.

def print_welcome(name: str, level: int) -> None:
    print(f"Welcome {name}! Since your level is {level}, you can do this. Keep going!")

print_welcome("King", 5)