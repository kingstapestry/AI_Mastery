import time
import inspect
from functools import wraps
from typing import Callable, Any


"""
LESSON 26: Decorators - One of Python's Most Powerful Features

Goal:
    Master decorators - functions that modify other functions.
    Used heavily in frameworks (Flask, FastAPI), logging, timing, caching, and authentication.
"""


# ==================== 1. BASIC DECORATOR ====================
def timer(func: Callable) -> Callable:
    """Simple decorator that measures how long a function takes to execute."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        result = func(*args, **kwargs)          # Call the original function
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️ {func.__name__} took {duration:.4f} seconds to run.")
        
        return result
    return wrapper


# ==================== 2. DECORATOR WITH ARGUMENTS ====================
def repeat(n: int):
    """Decorator that repeats the function execution n times."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            result = None
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


# ==================== 3. DECORATOR WITH METADATA PRESERVATION (@wraps) ====================
def debug(func: Callable) -> Callable:
    """Decorator that prints function name, arguments, and return value.
    Uses @wraps to preserve the original function's metadata (name, docstring)."""
    
    @wraps(func)                                # Very important - keeps original function info
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        # Get nice argument representation
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        argument_dict = dict(bound_args.arguments)
        
        print(f"[DEBUG] Calling {func_name} with args: {argument_dict}")
        
        result = func(*args, **kwargs)
        
        print(f"[DEBUG] {func_name} returned: {result}")
        return result
    return wrapper


# ==================== 4. CLASS-BASED DECORATOR ====================
class Singleton:
    """Class-based decorator that ensures only one instance of a class is created."""
    
    def __init__(self, cls):
        self._cls = cls
        self._instance = None
    
    def __call__(self, *args, **kwargs):
        """Called when you try to instantiate the class."""
        if self._instance is None:
            self._instance = self._cls(*args, **kwargs)
        return self._instance


# ==================== 5. PRACTICAL EXAMPLE - STACKING DECORATORS ====================
@debug
@timer
@repeat(n=3)
def process_data(data_id: int, mode: str = "fast") -> str:
    """Example function with multiple decorators stacked."""
    time.sleep(0.05)                            # Simulate work
    return f"Data-{data_id}-Processed-{mode.upper()}"


# ==================== 6. SINGLETON EXAMPLE ====================
@Singleton
class DatabaseConnection:
    """Example class that can only have one instance."""
    def __init__(self, connection_string: str):
        print(f"Connecting to database: {connection_string}")
        self.connection_string = connection_string


# ==================== 7. MAIN DEMO ====================
def main():
    print("=== Decorators Demo ===\n")
    
    print("--- Demo 1: Stacked Decorators ---")
    result = process_data(42, mode="secure")
    print(f"Final Result: {result}\n")
    
    print("--- Demo 2: Class-Based Decorator (Singleton) ---")
    db1 = DatabaseConnection("sqlite:////app.db")
    db2 = DatabaseConnection("postgresql://user:pass@localhost:5432/db")
    
    print(f"Are db1 and db2 the same instance? {db1 is db2}")
    print(f"Connection String: {db1.connection_string}")


if __name__ == "__main__":
    main()


# ==================== SUMMARY ====================
"""
Key Takeaways from Decorators:

- Decorators are functions that modify other functions or classes
- They are widely used for logging, timing, caching, authentication, etc.
- @wraps is important to preserve original function metadata
- You can stack multiple decorators (order matters)
- Class-based decorators are useful for more complex behavior (e.g. Singleton)
"""