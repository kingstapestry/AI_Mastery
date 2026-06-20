"""
my_package/calculator.py

Contains the Calculator class - a reusable component.
"""

class Calculator:
    """Professional Calculator class with proper documentation."""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b with proper error handling."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b