import pytest


"""
LESSON 28: Testing with pytest

Goal:
    Learn how to write proper tests — a critical professional skill.
    Good testing = confident code changes and fewer bugs.
"""


# ==================== 1. CODE TO TEST ====================
class Calculator:
    """Simple calculator class to demonstrate testing."""
    
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
        """Divide a by b. Raises ValueError if b is zero."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


# ==================== 2. BASIC TESTS ====================
def test_add():
    """Test the add method."""
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0
    assert calc.add(0, 0) == 0


def test_subtract():
    """Test the subtract method."""
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
    assert calc.subtract(3, 5) == -2


def test_multiply():
    """Test the multiply method."""
    calc = Calculator()
    assert calc.multiply(4, 5) == 20
    assert calc.multiply(-2, 3) == -6


def test_divide():
    """Test the divide method."""
    calc = Calculator()
    assert calc.divide(10, 2) == 5
    assert calc.divide(7, 2) == 3.5


# ==================== 3. EDGE CASE TESTS ====================
def test_divide_by_zero():
    """Test that dividing by zero raises ValueError."""
    calc = Calculator()
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calc.divide(10, 0)


# ==================== 4. FIXTURES ====================
@pytest.fixture
def calculator():
    """Fixture that provides a fresh Calculator instance for each test."""
    return Calculator()


def test_add_with_fixture(calculator):
    """Using fixture instead of creating Calculator manually."""
    assert calculator.add(10, 15) == 25


# ==================== 5. PARAMETRIZED TESTS ====================
@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
    (5.5, 2.5, 8.0)
])
def test_add_parametrized(a, b, expected):
    """Test add with multiple input combinations efficiently."""
    calc = Calculator()
    assert calc.add(a, b) == expected


# ==================== 6. MAIN (Run Tests) ====================
if __name__ == "__main__":
    # Run all tests with pytest
    pytest.main(["-v", __file__])   # -v = verbose output


# ==================== SUMMARY ====================
"""
Key Takeaways from Testing:

- Tests make your code reliable and easier to refactor
- pytest is the industry standard for Python testing
- Fixtures help avoid repetitive setup code
- Parametrized tests let you test multiple cases efficiently
- Always test edge cases (like division by zero)
"""