"""
main.py

Demonstrates how to use your custom package.
"""

from my_package import Calculator, format_number, log_message

def main():
    print("=== Using Custom Package Demo ===\n")
    
    calc = Calculator()
    
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 / 2 = {calc.divide(10, 2)}")
    
    formatted = format_number(123.4567, decimals=2)
    print(f"Formatted number: {formatted}")
    
    log_message("This is an info message")
    log_message("This is a warning", level="warning")
    
    print("\n=== Package Demo Completed ===")


if __name__ == "__main__":
    main()