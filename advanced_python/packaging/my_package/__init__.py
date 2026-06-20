"""
my_package/__init__.py

This file makes the folder a Python package.
It also controls what gets imported when someone does `import my_package`.
"""

__version__ = "0.1.0"

# Make important classes/functions available directly
from .calculator import Calculator
from .utils import format_number, log_message

__all__ = ["Calculator", "format_number", "log_message"]