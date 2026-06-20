from typing import Union, Optional


"""
LESSON 33: Documentation & Code Quality

Goal:
    Learn how to write professional, well-documented code that is easy to understand,
    maintain, and use by others — a critical skill at top AI companies.
"""


# ==================== 1. WELL-DOCUMENTED CLASS ====================
class Calculator:
    """
    A simple calculator class demonstrating professional documentation standards.
    
    This class provides basic arithmetic operations with full type hints and
    comprehensive Google-style docstrings.
    
    Attributes:
        precision (int): Number of decimal places for results (default: 4)
    """
    
    def __init__(self, precision: int = 4):
        """
        Initialize the calculator.
        
        Args:
            precision (int): Number of decimal places to round results to.
        """
        self.precision = precision
    
    def add(self, a: Union[int, float], b: Union[int, float]) -> float:
        """
        Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b, rounded to the set precision.
        """
        return round(a + b, self.precision)
    
    def subtract(self, a: Union[int, float], b: Union[int, float]) -> float:
        """
        Subtract b from a.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Result of a - b.
        """
        return round(a - b, self.precision)
    
    def multiply(self, a: Union[int, float], b: Union[int, float]) -> float:
        """
        Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b.
        """
        return round(a * b, self.precision)
    
    def divide(self, a: Union[int, float], b: Union[int, float]) -> float:
        """
        Divide a by b.
        
        Args:
            a: Numerator
            b: Denominator
            
        Returns:
            Result of a / b.
            
        Raises:
            ValueError: If b is zero.
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return round(a / b, self.precision)


# ==================== 2. UTILITY FUNCTIONS WITH DOCSTRINGS ====================
def format_result(value: float, decimals: Optional[int] = None) -> str:
    """
    Format a number with specified decimal places.
    
    Args:
        value: The number to format
        decimals: Number of decimal places (uses calculator default if None)
        
    Returns:
        Formatted string representation of the number.
    """
    if decimals is None:
        decimals = 4
    return f"{value:.{decimals}f}"


# ==================== 3. MAIN DEMO WITH USAGE EXAMPLES ====================
def main():
    """
    Demonstrates usage of the documented Calculator class.
    """
    print("=== Documentation & Code Quality Demo ===\n")
    
    calc = Calculator(precision=6)
    
    print("Addition:", calc.add(3.14159, 2.71828))
    print("Division:", calc.divide(100, 7))
    
    try:
        calc.divide(10, 0)
    except ValueError as e:
        print(f"Caught error: {e}")
    
    print("\nFormatted Result:", format_result(calc.multiply(2.5, 4.8), decimals=3))
    
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    main()


# ==================== 4. CODE QUALITY NOTES ====================
"""
CODE QUALITY BEST PRACTICES:

1. Linting:
   - Black: Code formatter (pip install black)
   - Flake8: Style checker (pip install flake8)
   - mypy: Type checker (pip install mypy)

2. Documentation Styles:
   - Google Style (used above) - very readable
   - NumPy Style - good for scientific code
   - Sphinx Style - for auto-generated docs

3. Why This Matters:
   - Makes code easier to understand and maintain
   - Required for collaboration and open source
   - Shows professionalism to employers
"""