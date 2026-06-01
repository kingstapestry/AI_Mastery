# This is a comment (Python ignores everything after #)

print("Hello, I am King and I will become an AI Master!")

# Variables - store data (without type hints (python is dynamically typed))
name = "King"           # Text (string)
city = "Vancouver"
age = 23                # Number (integer)
height = 5.9            # Decimal number (float)
is_learning_ai = True   # True or False (boolean)

# Or if you wish to have type hints
# name: str = "King"
# city: str = "Vancouver"
# age: int = 23
# height: float = 5.9
# is_learning_ai: bool = True

print("My name is", name)
print("I am", age, "years old")
print("I am from", city)

if is_learning_ai is True:
    print(f"Learning AI: {is_learning_ai}")
else:
    print(f"Learning AI: {is_learning_ai}")