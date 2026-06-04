name = input("What is your name? ")     # Ask for input from user
age = int(input("How old are you? "))   # int() converts text to number 

if age >= 18:
    print(name, "you are an adult. Good for AI journey!")
elif age >= 13:
    print("Teenager phase - perfect time to learn!")
elif age < 10:
    print("What are you even here for?")
else:
    print("Young! Start early 🔥")