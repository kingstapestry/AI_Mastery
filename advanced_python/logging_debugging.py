import logging
import traceback
import sys
from functools import wraps
from typing import Callable, Any


"""
LESSON 29: Logging, Debugging & Error Handling

Goal:
    Learn professional practices for making your code robust, observable, and maintainable.
    These skills are essential for production AI/ML systems.
"""


# ==================== 1. BASIC LOGGING SETUP ====================
# Configure logging at the start of your application
logging.basicConfig(
    level=logging.INFO,                     # Change to DEBUG for development
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


# ==================== 2. ADVANCED LOGGER SETUP ====================
def setup_logger(name: str = __name__):
    """Create a professional logger that outputs to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(console_format)
    
    # File Handler
    file_handler = logging.FileHandler(f"{name}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s')
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


logger = setup_logger("app")


# ==================== 3. DECORATOR FOR ERROR HANDLING & LOGGING ====================
def log_errors(func: Callable) -> Callable:
    """
    Decorator that catches exceptions, logs them, and re-raises.
    Very useful for production code.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())    # Full stack trace
            raise                                   # Re-raise the exception
    return wrapper


# ==================== 4. ROBUST ERROR HANDLING EXAMPLE ====================
@log_errors
def safe_divide(a: float, b: float) -> float:
    """Safely divides two numbers with proper error handling."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# ==================== 5. PRACTICAL EXAMPLE ====================
@log_errors
def process_numbers(numbers: list) -> dict:
    """Example function that processes a list of numbers with full logging."""
    logger.info(f"Processing {len(numbers)} numbers")
    
    total = 0
    valid_count = 0
    
    for num in numbers:
        try:
            if not isinstance(num, (int, float)):
                raise TypeError(f"Invalid type: {type(num)}")
            total += num
            valid_count += 1
        except Exception as e:
            logger.warning(f"Skipping invalid number {num}: {e}")
    
    if valid_count == 0:
        logger.error("No valid numbers processed!")
        return {"total": 0, "count": 0, "average": 0}
    
    average = total / valid_count
    logger.info(f"Processed {valid_count} valid numbers. Average: {average:.2f}")
    
    return {
        "total": total,
        "count": valid_count,
        "average": average
    }


# ==================== 6. MAIN DEMO ====================
def main():
    print("=== Logging, Debugging & Error Handling Demo ===\n")
    
    # Basic logging examples
    logger.debug("This is a debug message (only visible in DEBUG mode)")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\n--- Safe Division Examples ---")
    try:
        print(safe_divide(10, 2))
        print(safe_divide(10, 0))           # This will be caught and logged
    except Exception as e:
        print(f"Caught error: {e}")
    
    print("\n--- Process Numbers Example ---")
    data = [1, 2, "invalid", 4, None, 6]
    result = process_numbers(data)
    print(f"Final Result: {result}")
    
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    main()


# ==================== SUMMARY ====================
"""
Key Takeaways:

- Logging is much better than print() for production code
- Use different levels: DEBUG (development), INFO, WARNING, ERROR, CRITICAL
- Decorators can automatically add logging and error handling
- Always catch and log exceptions with full stack traces (traceback)
- Good logging = easier debugging and monitoring in production
"""